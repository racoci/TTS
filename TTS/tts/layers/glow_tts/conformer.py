import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F

import numpy as np
from TTS.tts.layers.glow_tts.glow import LayerNorm

class Swish(nn.Module):
    """ Implementation for Swish activation Function: https://arxiv.org/abs/1710.05941 """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)

class Mish(nn.Module):
    """ Implementation for Mish activation Function: https://www.bmvc2020-conference.com/assets/papers/0928.pdf"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class FFN(nn.Module):
    """Positionwise fully-connected feed-forward neural network (FFN) layer.
    Args:
        in_out_dim (int): input and output dimension
        hidden_channels (int): hidden dimension
        dropout (float): dropout probability
        activation (str): non-linear activation function
    """

    def __init__(self, in_out_dim, hidden_channels, activation='swish', dropout=0.1):

        super().__init__()
        self.dense1 = nn.Linear(in_out_dim, hidden_channels)
        self.dense2 = nn.Linear(hidden_channels, in_out_dim)

        self.dropout = nn.Dropout(p=dropout)
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'swish':
            self.activation = Swish()
        elif activation == 'mish':
            self.activation = Mish()
        else:
            raise NotImplementedError(activation)       

        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)

    def forward(self, x):
        """
        Args:
            x (FloatTensor): [B, T, hidden_channels]
        Returns:
            x (FloatTensor): [B, T, hidden_channels]
        """
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class XLPositionalEmbedding(nn.Module):
    """Positional embedding for TransformerXL."""
    def __init__(self, channels, dropout):

        super().__init__()

        self.channels = channels
        inv_freq = 1 / (10000 ** (torch.arange(0.0, channels, 2.0) / channels))
        self.register_buffer("inv_freq", inv_freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xs, mlen=0, clamp_len=-1, zero_center_offset=False):
        """Forward pass.
        Args:
            xs (FloatTensor):[B, L, channels]
            mlen (int); length of memory
            clamp_len (int):
            zero_center_offset (bool):
        Returns:
            pos_emb (LongTensor):[L, 1, channels]
        """
        if zero_center_offset:
            pos_idxs = torch.arange(mlen - 1, -xs.size(1) - 1, -1.0, dtype=torch.float, device=xs.device)
        else:
            pos_idxs = torch.arange(mlen + xs.size(1) - 1, -1, -1.0, dtype=torch.float, device=xs.device)

        # truncate by maximum length
        if clamp_len > 0:
            pos_idxs.clamp_(max=clamp_len)

        # outer product
        sinusoid_inp = torch.einsum("i,j->ij", pos_idxs, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = self.dropout(pos_emb)
        return pos_emb.unsqueeze(1)

class ConformerConvModule(nn.Module):
    """A single convolution module for the Conformer.
    Args:
        in_out_dim (int): input and output dimension
        kernel_size (int): kernel size in depthwise convolution
    """

    def __init__(self, in_out_dim, kernel_size=2, activation='swish', causal=False):
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
        elif activation == 'swish':
            self.activation = Swish()
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
            x (FloatTensor): [B, T, in_out_dim]
        Returns:
            x (FloatTensor): [B, T, in_out_dim]
        """
        x = x.transpose(2, 1).contiguous()  # [B, C, T]
        if self.padding is not None:
            x = self.padding(x)
            x = x[:, :, :-(self.kernel_size - 1)]

        x = self.pointwise_layer(x)  # [B, 2 * C, T]
        x = x.transpose(2, 1)  # [B, T, 2 * C]
        x = F.glu(x)  # [B, T, C]+

        x = x.transpose(2, 1).contiguous()  # [B, C, T]

        x = self.depthwise_layer(x)  # [B, C, T]

        x = self.bn(x)
        x = self.activation(x)
        x = self.pointwise_layer2(x)  # [B, C, T]

        x = x.transpose(2, 1).contiguous()  # [B, T, C]
        return x

class RelativeMultiheadAttentionMechanism(nn.Module):
    """Relative multi-head attention layer for TransformerXL.
    Args:
        channels (int): dimension of key, query, attention space and output
        n_heads (int): number of heads
        dropout (float): dropout probability for attenion weights
        bias (bool): use bias term in linear layers
    """

    def __init__(self, channels, n_heads, dropout,
                 bias=False, ):

        super().__init__()

        assert channels % n_heads == 0
        self.d_k = channels // n_heads
        self.n_heads = n_heads
        self.scale = math.sqrt(self.d_k)

        self.dropout_attn = nn.Dropout(p=dropout)

        self.w_key = nn.Linear(channels, channels, bias=bias)
        self.w_value = nn.Linear(channels, channels, bias=bias)
        self.w_query = nn.Linear(channels, channels, bias=bias)
        self.w_out = nn.Linear(channels, channels, bias=bias)

        self.w_pos = nn.Linear(channels, channels, bias=bias)


        nn.init.xavier_uniform_(self.w_key.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_value.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w_query.weight, gain=1 / math.sqrt(2))
        if bias:
            nn.init.constant_(self.w_key.bias, 0.)
            nn.init.constant_(self.w_value.bias, 0.)
            nn.init.constant_(self.w_query.bias, 0.)

        nn.init.xavier_uniform_(self.w_out.weight)
        if bias:
            nn.init.constant_(self.w_out.bias, 0.)

        
        nn.init.xavier_uniform_(self.w_pos.weight)
        if bias:
            nn.init.constant_(self.w_pos.bias, 0.)

    def _rel_shift(self, x):
        """Calculate relative positional attention efficiently.
        Args:
            x (FloatTensor): [B, qlen, klen, H]
        Returns:
            x_shifted (FloatTensor): [B, qlen, klen, H]
        """
        bs, qlen, klen, n_heads = x.size()
        # [qlen, klen, B, H] -> [B, qlen, klen, H]
        x = x.permute(1, 2, 0, 3).contiguous().view(qlen, klen, bs * n_heads)

        zero_pad = x.new_zeros((qlen, 1, bs * n_heads))
        x_shifted = (torch.cat([zero_pad, x], dim=1)
                      .view(klen + 1, qlen, bs * n_heads)[1:]
                      .view_as(x))
        return x_shifted.view(qlen, klen, bs, n_heads).permute(2, 0, 1, 3)

    def forward(self, key, query, pos_embs, mask, u_bias=None, v_bias=None):
        """Forward pass.
        Args:
            key (FloatTensor): [B, mlen+qlen, kdim]
            pos_embs (LongTensor): [qlen, 1, hidden_channels]
            mask (ByteTensor): [B, qlen, mlen+qlen]
            u_bias (nn.Parameter): [H, d_k]
            v_bias (nn.Parameter): [H, d_k]
        Returns:
            cv (FloatTensor): [B, qlen, vdim]
            aw (FloatTensor): [B, H, qlen, mlen+qlen]
        """
        bs, qlen = query.size()[:2]
        mlen = key.size(1) - qlen

        if mask is not None:
            mask = mask.squeeze(1).unsqueeze(-1).repeat([1, 1, 1, self.n_heads])
            assert mask.size() == (bs, qlen, mlen + qlen, self.n_heads), \
                (mask.size(), (bs, qlen, mlen + qlen, self.n_heads))

        k = self.w_key(key).view(bs, -1, self.n_heads, self.d_k)  # [B, mlen+qlen, H, d_k]
        v = self.w_value(key).view(bs, -1, self.n_heads, self.d_k)  # [B, mlen+qlen, H, d_k]
        q = self.w_query(key[:, -qlen:]).view(bs, -1, self.n_heads, self.d_k)  # [B, qlen, H, d_k]

        _pos_embs = self.w_pos(pos_embs)

        _pos_embs = _pos_embs.view(-1, self.n_heads, self.d_k)  # [mlen+qlen, H, d_k]

        # content-based attention term: (a) + (c)
        if u_bias is not None:
            AC = torch.einsum("bihd,bjhd->bijh", ((q + u_bias[None, None]), k))  # [B, qlen, mlen+qlen, H]
        else:
            AC = torch.einsum("bihd,bjhd->bijh", (q, k))  # [B, qlen, mlen+qlen, H]

        # position-based attention term: (b) + (d)
        if v_bias is not None:
            BD = torch.einsum("bihd,jhd->bijh", ((q + v_bias[None, None]), _pos_embs))  # [B, qlen, mlen+qlen, H]
        else:
            BD = torch.einsum("bihd,jhd->bijh", (q, _pos_embs))  # [B, qlen, mlen+qlen, H]

        # Compute positional attention efficiently
        BD = self._rel_shift(BD)

        # the attention is the sum of content-based and position-based attention
        e = (AC + BD) / self.scale  # [B, qlen, mlen+qlen, H]

        # Compute attention weights
        if mask is not None:
            NEG_INF = float(np.finfo(torch.tensor(0, dtype=e.dtype).numpy().dtype).min)
            e = e.masked_fill_(mask == 0, NEG_INF)  # [B, qlen, mlen+qlen, H]

        aw = torch.softmax(e, dim=2)
        aw = self.dropout_attn(aw)  # [B, qlen, mlen+qlen, H]

        cv = torch.einsum("bijh,bjhd->bihd", (aw, v))  # [B, qlen, H, d_k]
        cv = cv.contiguous().view(bs, -1, self.n_heads * self.d_k)  # [B, qlen, H * d_k]
        cv = self.w_out(cv)
        aw = aw.permute(0, 3, 1, 2)  # [B, H, qlen, mlen+qlen]
        return cv, aw


class ConformerBlock(nn.Module):
    """A single block of the Conformer.
    Args:
        hidden_channels (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        lnorm_eps (float): epsilon parameter for layer normalization
        ffn_activation (str): nonolinear function for PositionwiseFeedForward
        unidirectional (bool): pad right context for unidirectional encoding
    """

    def __init__(self, hidden_channels, d_ff=1024, n_heads=4, kernel_size=3,
                 dropout=0.1, dropout_att=0.1,
                 lnorm_eps=1e-05, ffn_activation='swish', unidirectional=False):
        super(ConformerBlock, self).__init__()

        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first half position-wise feed-forward
        self.norm1 = nn.LayerNorm(hidden_channels, eps=lnorm_eps)
        self.feed_forward_macaron = FFN(hidden_channels, d_ff, ffn_activation, dropout)

        # self-attention
        self.norm2 = nn.LayerNorm(hidden_channels, eps=lnorm_eps)
        self.self_attn = RelativeMultiheadAttentionMechanism(channels=hidden_channels,
                                n_heads=n_heads,
                                dropout=dropout_att)

        # conv module
        self.norm3 = nn.LayerNorm(hidden_channels, eps=lnorm_eps)
        self.conv = ConformerConvModule(hidden_channels, kernel_size, causal=unidirectional)

        # second half position-wise feed-forward
        self.norm4 = nn.LayerNorm(hidden_channels, eps=lnorm_eps)
        self.feed_forward = FFN(hidden_channels, d_ff, ffn_activation, dropout)
        self.norm5 = nn.LayerNorm(hidden_channels, eps=lnorm_eps)

        self.dropout = nn.Dropout(dropout)

        self.reset_visualization()

    @property
    def xx_aws(self):
        return self._xx_aws

    def reset_visualization(self):
        self._xx_aws = None

    def forward(self, x, xx_mask=None, pos_embs=None, u_bias=None, v_bias=None):
        """Conformer encoder layer definition.
        Args:
            x (FloatTensor): [B, T, hidden_channels]
            xx_mask (ByteTensor): [B, T (query), T (key)]
            pos_embs (LongTensor): [L, 1, hidden_channels]
            u_bias (FloatTensor): global parameter for relative positional encoding
            v_bias (FloatTensor): global parameter for relative positional encoding
        Returns:
            x (FloatTensor): [B, T, hidden_channels]
        """
        self.reset_visualization()
        # first half FFN
        residual = x
        x = self.norm1(x)
        x = self.feed_forward_macaron(x)
        x = self.fc_factor * self.dropout(x) + residual  # Macaron FFN

        # self-attention w/ relative positional encoding
        residual = x
        x = self.norm2(x)
        x, self._xx_aws = self.self_attn(x, x, pos_embs, xx_mask, u_bias, v_bias)
        x = self.dropout(x) + residual
        # conv
        residual = x
        x = self.norm3(x)
        x = self.conv(x)
        x = self.dropout(x) + residual

        # second half FFN
        residual = x
        x = self.norm4(x)
        x = self.feed_forward(x)
        x = self.fc_factor * self.dropout(x) + residual  # Macaron FFN
        x = self.norm5(x)  # this is important for performance

        return x


class Conformer(nn.Module):
    """Implementation of Conformers: https://arxiv.org/pdf/2005.08100.pdf"""
    def __init__(self,
                 hidden_channels,
                 d_ff,
                 num_heads,
                 num_layers,
                 kernel_size=3,
                 dropout_p=0.1, positional_embedding=True):
        super().__init__()

        self.num_layers = num_layers
        self.positional_embedding = positional_embedding
        if positional_embedding:
            self.scale = math.sqrt(hidden_channels)
            self.pos_emb = XLPositionalEmbedding(hidden_channels, dropout_p) # transformer XL like Positional Embedding
        

        # u_bias and v_bias are global parameters
        self.u_bias = nn.Parameter(torch.Tensor(num_heads, hidden_channels // num_heads))
        self.v_bias = nn.Parameter(torch.Tensor(num_heads, hidden_channels // num_heads))
           
        self.conformers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.conformers.append(ConformerBlock(hidden_channels, d_ff=d_ff, n_heads=num_heads, kernel_size=kernel_size,
                 dropout=dropout_p, dropout_att=0.0, lnorm_eps=1e-05, ffn_activation='swish', unidirectional=False, ))

    def forward(self, x, x_mask):
        x = x.transpose(1,2)
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x_mask = x_mask.transpose(1,2)
        if self.positional_embedding:
            x = x * self.scale
            pos_embs = self.pos_emb(x, zero_center_offset=True)  # NOTE: no clamp_len for streaming
        else:
            pos_embs = None

        for i in range(self.num_layers):
            x = x * x_mask
            x = self.conformers[i](x, attn_mask, pos_embs=pos_embs, u_bias=self.u_bias, v_bias=self.v_bias)
        x = x * x_mask
        x = x.transpose(1,2)
        return x
