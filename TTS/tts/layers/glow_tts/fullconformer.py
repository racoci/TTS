import math
import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.glow_tts.glow import LayerNorm



class Mish(nn.Module):
    """ Implementation for Mish activation Function: https://www.bmvc2020-conference.com/assets/papers/0928.pdf"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class RelativePositionMultiHeadAttention(nn.Module):
    """Implementation of Relative Position Encoding based on
    https://arxiv.org/pdf/1809.04281.pdf
    """
    def __init__(self,
                 channels,
                 out_channels,
                 num_heads,
                 rel_attn_window_size=None,
                 heads_share=True,
                 dropout_p=0.,
                 input_length=None,
                 proximal_bias=False,
                 proximal_init=False):
        super().__init__()
        assert channels % num_heads == 0, " [!] channels should be divisible by num_heads."
        # class attributes
        self.channels = channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.rel_attn_window_size = rel_attn_window_size
        self.heads_share = heads_share
        self.input_length = input_length
        self.proximal_bias = proximal_bias
        self.dropout_p = dropout_p
        self.attn = None
        # query, key, value layers
        self.k_channels = channels // num_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        # output layers
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_p)
        # relative positional encoding layers
        if rel_attn_window_size is not None:
            num_heads_rel = 1 if heads_share else num_heads
            rel_stddev = self.k_channels**-0.5
            emb_rel_k = nn.Parameter(
                torch.randn(num_heads_rel, rel_attn_window_size * 2 + 1,
                            self.k_channels) * rel_stddev)
            emb_rel_v = nn.Parameter(
                torch.randn(num_heads_rel, rel_attn_window_size * 2 + 1,
                            self.k_channels) * rel_stddev)
            self.register_parameter('emb_rel_k', emb_rel_k)
            self.register_parameter('emb_rel_v', emb_rel_v)

        # init layers
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        # proximal bias
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.num_heads, self.k_channels,
                           t_t).transpose(2, 3)
        key = key.view(b, self.num_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.num_heads, self.k_channels,
                           t_s).transpose(2, 3)
        # compute raw attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.k_channels)
        # relative positional encoding
        if self.rel_attn_window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            # get relative key embeddings
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(
                rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        # proximan bias
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attn_proximity_bias(t_s).to(
                device=scores.device, dtype=scores.dtype)
        # attention score masking
        if mask is not None:
            # add small value to prevent oor error.
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.input_length is not None:
                block_mask = torch.ones_like(scores).triu(
                    -1 * self.input_length).tril(self.input_length)
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        # attention score normalization
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        # apply dropout to attention weights
        p_attn = self.dropout(p_attn)
        # compute output
        output = torch.matmul(p_attn, value)
        # relative positional encoding for values
        if self.rel_attn_window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(
            b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    @staticmethod
    def _matmul_with_relative_values(p_attn, re):
        """
        Args:
            p_attn (Tensor): attention weights.
            re (Tensor): relative value embedding vector. (a_(i,j)^V)

        Shapes:
            p_attn: [B, H, T, V]
            re: [H or 1, V, D]
            logits: [B, H, T, D]
        """
        logits = torch.matmul(p_attn, re.unsqueeze(0))
        return logits

    @staticmethod
    def _matmul_with_relative_keys(query, re):
        """
        Args:
            query (Tensor): batch of query vectors. (x*W^Q)
            re (Tensor): relative key embedding vector. (a_(i,j)^K)

        Shapes:
            query: [B, H, T, D]
            re: [H or 1, V, D]
            logits: [B, H, T, V]
        """
        # logits = torch.einsum('bhld, kmd -> bhlm', [query, re.to(query.dtype)])
        logits = torch.matmul(query, re.unsqueeze(0).transpose(-2, -1))
        return logits

    def _get_relative_embeddings(self, relative_embeddings, length):
        """Convert embedding vestors to a tensor of embeddings
        """
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.rel_attn_window_size + 1), 0)
        slice_start_position = max((self.rel_attn_window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                                              slice_start_position:
                                                              slice_end_position]
        return used_relative_embeddings

    @staticmethod
    def _relative_position_to_absolute_position(x):
        """Converts tensor from relative to absolute indexing for local attention.
        Args:
            x: [B, D, length, 2 * length - 1]
        Returns:
            A Tensor of shape [B, D, length, length]
        """
        batch, heads, length, _ = x.size()
        # Pad to shift from relative to absolute indexing.
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        # Pad extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1,
                               2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    @staticmethod
    def _absolute_position_to_relative_position(x):
        """
        x: [B, H, T, T]
        ret: [B, H, T, 2*T-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    @staticmethod
    def _attn_proximity_bias(length):
        """Produce an attention mask that discourages distant
        attention values.
        Args:
            length (int): an integer scalar.
        Returns:
            a Tensor with shape [1, 1, length, length]
        """
        # L
        r = torch.arange(length, dtype=torch.float32)
        # L x L
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        # scale mask values
        diff = -torch.log1p(torch.abs(diff))
        # 1 x 1 x L x L
        return diff.unsqueeze(0).unsqueeze(0)

class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_channels,
                 kernel_size,
                 dropout_p=0.,
                 activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.activation = activation

        self.conv_1 = nn.Conv1d(in_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels,
                                out_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout_p)
                # init layers
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
    def forward(self, x, x_mask):
        # print(x_mask)
        x = self.conv_1(x * x_mask)
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        elif self.activation == "mish":
            x = x * torch.tanh(F.softplus(x))
        else:
            x = torch.relu(x)

        x = self.dropout(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask



class ConformerConvModule(nn.Module):
    """A single convolution module for the Conformer.
    Args:
        in_out_dim (int): input and output dimension
        kernel_size (int): kernel size in depthwise convolution
    """

    def __init__(self, in_out_dim, kernel_size=2, activation='mish', causal=False):
        super().__init__()
        self.kernel_size = kernel_size

        assert (self.kernel_size  - 1) % 2 == 0, 'kernel_size must be the odd number.'

        if causal:
            self.padding = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        else:
            self.padding = None

        self.pointwise_layer = nn.Conv1d(in_channels=in_out_dim,
                                         out_channels=in_out_dim * 2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.depthwise_layer = nn.Conv1d(in_channels=in_out_dim,
                                        out_channels=in_out_dim,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=(kernel_size - 1) // 2,
                                        groups=in_out_dim)
        self.bn = nn.BatchNorm1d(in_out_dim)

        self.pointwise_layer2 = nn.Conv1d(in_channels=in_out_dim,
                                         out_channels=in_out_dim,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)               
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'mish':
            self.activation = Mish()
        else:
            raise NotImplementedError(activation)

        nn.init.xavier_uniform_(self.pointwise_layer.weight)
        nn.init.xavier_uniform_(self.depthwise_layer.weight)
        nn.init.xavier_uniform_(self.pointwise_layer2.weight)

    def forward(self, x):
        """
        Args:
            x (FloatTensor): [B, in_out_dim, T]
        Returns:
            x (FloatTensor): [B, in_out_dim, T]
        """
        if self.padding is not None:
            x = self.padding(x)
            x = x[:, :, :-(self.kernel_size - 1)]

        x = self.pointwise_layer(x)  # [B, 2 * C, T]
        x = x.transpose(2, 1)  # [B, T, 2 * C]
        x = F.glu(x)  # [B, T, C]

        x = x.transpose(2, 1).contiguous()  # [B, C, T]

        x = self.depthwise_layer(x)  # [B, C, T]

        x = self.bn(x)
        x = self.activation(x)
        x = self.pointwise_layer2(x)  # [B, C, T]

        return x

class FullConformerBlock(nn.Module):
    """A single block of the Full Convolutional Conformer (Conformer paper: https://arxiv.org/pdf/2005.08100.pdf).
    Improviments:
    We do not use TransformerXL embedding position. We chose to use Relative Positional MultiHead Attention based on https://arxiv.org/pdf/1809.04281.pdf
    In addition, we replaced the fully connected FFN by a FFN based on convolutional layers (proposed at: https://www.aaai.org/ojs/index.php/AAAI/article/view/4642/4520)
   
    Args:
        hidden_channels (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        num_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
    """

    def __init__(self, hidden_channels, d_ff=1024, filter_channels=256, num_heads=4, kernel_size=3,
                 dropout=0.1, dropout_att=0.1, ffn_activation='mish', rel_attn_window_size=None, input_length=None):
        super(FullConformerBlock, self).__init__()

        self.num_heads = num_heads
        self.fc_factor = 0.5

        # first half position-wise feed-forward
        #self.norm1 = LayerNorm(hidden_channels)
        self.feed_forward_macaron = FFN(hidden_channels, d_ff, filter_channels, kernel_size, dropout_p=dropout, activation=ffn_activation)
        # self-attention
        self.norm2 = LayerNorm(hidden_channels)
        self.self_attn = RelativePositionMultiHeadAttention(hidden_channels, hidden_channels, num_heads, rel_attn_window_size=rel_attn_window_size, dropout_p=dropout_att, input_length=input_length)
        # conv module
        self.norm3 = LayerNorm(hidden_channels)
        self.conv = ConformerConvModule(hidden_channels, kernel_size, causal=True)

        # second half position-wise feed-forward
        self.norm4 = LayerNorm(hidden_channels)
        self.feed_forward = FFN(hidden_channels, d_ff, filter_channels, kernel_size, dropout_p=dropout, activation=ffn_activation)
        self.norm5 = LayerNorm(hidden_channels)

        self.dropout = nn.Dropout(dropout)

        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, x, attn_mask=None, x_mask=None):
        """Conformer encoder layer definition.
        Args:
            x (FloatTensor): [B, T, hidden_channels]
            attn_mask (ByteTensor): [B, T (query), T (key)]
        Returns:
            x (FloatTensor): [B, T, hidden_channels]
        """
        self.reset_visualization()
        # first half FFN
        residual = x
        # x = self.norm1(x)
        x = self.feed_forward_macaron(x, x_mask)
        x = self.fc_factor * self.dropout(x) + residual  # Macaron FFN

        # self-attention w/ relative positional encoding
        residual = x
        x = self.norm2(x)
        x = self.self_attn(x, x, attn_mask)
        x = self.dropout(x) + residual
        # conv
        residual = x
        x = self.norm3(x)
        x = self.conv(x)
        x = self.dropout(x) + residual

        # second half FFN
        residual = x
        x = self.norm4(x)
        x = self.feed_forward(x, x_mask)
        x = self.fc_factor * self.dropout(x) + residual  # Macaron FFN
        x = self.norm5(x)  # this is important for performance

        return x

class FullConformer(nn.Module):
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 num_heads,
                 num_layers,
                 kernel_size=1,
                 dropout_p=0.,
                 rel_attn_window_size=None,
                 input_length=None):
        super(FullConformer, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.rel_attn_window_size = rel_attn_window_size

        self.conformers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.conformers.append(FullConformerBlock(hidden_channels, d_ff=filter_channels, num_heads=num_heads, kernel_size=kernel_size,
                 dropout=dropout_p, dropout_att=dropout_p, ffn_activation='mish', rel_attn_window_size=rel_attn_window_size, input_length=input_length)) 

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.num_layers):
            x = x * x_mask
            x = self.conformers[i](x, attn_mask, x_mask)
        x = x * x_mask
        return x
