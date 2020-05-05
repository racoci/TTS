import copy
import torch
from math import sqrt
from torch import nn
from TTS.layers.tacotron2 import Encoder, Decoder, Postnet
from TTS.utils.generic_utils import sequence_mask
from TTS.layers.gst_layers import GST

# TODO: match function arguments with tacotron
class Tacotron2(nn.Module):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r,
                 speaker_embedding_dim=512,
                 postnet_output_dim=80,
                 decoder_output_dim=80,
                 attn_type='original',
                 attn_win=False,
                 gst=False,
                 attn_norm="softmax",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 attn_K=5,
                 separate_stopnet=True,
                 bidirectional_decoder=False):
        super(Tacotron2, self).__init__()
        self.postnet_output_dim = postnet_output_dim
        self.decoder_output_dim = decoder_output_dim
        self.gst = gst
        self.num_speakers = num_speakers
        self.n_frames_per_step = r
        self.bidirectional_decoder = bidirectional_decoder

        gst_embedding_dim = 256 if self.gst else 0
        decoder_dim = 512+speaker_embedding_dim+gst_embedding_dim  if num_speakers > 1 else 512
        encoder_dim = 512 if num_speakers > 1 else 512
        proj_speaker_dim = 80 if num_speakers > 1 else 0
        # embedding layer
        self.embedding = nn.Embedding(num_chars, 512, padding_idx=0)
        std = sqrt(2.0 / (num_chars + 512))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        if num_speakers > 1:
            if self.gst:
                self.speaker_project_mel = nn.Sequential(
                    nn.Linear(speaker_embedding_dim+gst_embedding_dim, proj_speaker_dim), nn.Tanh())
            else:
                self.speaker_project_mel = nn.Sequential(
                    nn.Linear(speaker_embedding_dim, proj_speaker_dim), nn.Tanh())

            self.speaker_embeddings = None
            self.speaker_embeddings_projected = None
        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(decoder_dim, self.decoder_output_dim, r, attn_type, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, attn_K, separate_stopnet, proj_speaker_dim)
        if self.bidirectional_decoder:
            self.decoder_backward = copy.deepcopy(self.decoder)
        self.postnet = Postnet(self.postnet_output_dim)
        # global style token layers
        if self.gst:
            self.gst_layer = GST(num_mel=80,
                                 num_heads=8,
                                 num_style_tokens=10,
                                 embedding_dim=gst_embedding_dim)

    def _init_states(self):
        self.speaker_embeddings = None
        self.speaker_embeddings_projected = None

    def compute_gst(self, inputs, mel_specs):
        gst_outputs = self.gst_layer(mel_specs)
        inputs = self._concat_speaker_embedding(inputs, gst_outputs)
        #inputs = self._add_speaker_embedding(inputs, gst_outputs)
        return  inputs, gst_outputs

    def forward(self, characters, text_lengths, mel_specs=None, speaker_embeddings=None):
        self._init_states()
        # compute mask for padding
        mask = sequence_mask(text_lengths).to(characters.device)
        embedded_inputs = self.embedding(characters).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        if self.gst:
            # B x gst_dim
            encoder_outputs, gts_embedding = self.compute_gst(encoder_outputs, mel_specs)
        if speaker_embeddings is not None:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs.transpose(0, 1), speaker_embeddings).transpose(0, 1)
            if self.gst:
                self.speaker_embeddings_projected = self.speaker_project_mel(
                    self._concat_speaker_embedding(gts_embedding.transpose(0, 1), speaker_embeddings)).transpose(0, 1).squeeze(1)
            else:
                self.speaker_embeddings_projected = self.speaker_project_mel(
                        speaker_embeddings).squeeze(1)
        decoder_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask,
            self.speaker_embeddings_projected)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(
            decoder_outputs, postnet_outputs, alignments)
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_inference(mel_specs, encoder_outputs, mask)
            return decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments



    @torch.no_grad()
    def inference(self, characters, speaker_embedding=None, style_mel=None):
        embedded_inputs = self.embedding(characters).transpose(1, 2)
        self._init_states()
        encoder_outputs = self.encoder.inference(embedded_inputs)
        if self.gst and style_mel is not None:
            encoder_outputs, gst_embedding = self.compute_gst(encoder_outputs, style_mel)
        if speaker_embedding is not None:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs.transpose(0, 1), speaker_embedding).transpose(0, 1)
            
            if self.gst and style_mel is not None:
                self.speaker_embeddings_projected = self.speaker_project_mel(
                        self._concat_speaker_embedding(gst_embedding.transpose(0, 1), speaker_embedding)).transpose(0, 1).squeeze(1)
            else:    
                self.speaker_embeddings_projected = self.speaker_project_mel(
                        speaker_embedding).squeeze(1)
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs, self.speaker_embeddings_projected)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens

    def inference_truncated(self, text, speaker_embedding=None, style_mel=None):
        """
        Preserve model states for continuous inference
        """
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference_truncated(embedded_inputs)
        if self.gst and style_mel is not None:
            encoder_outputs, gst_embedding = self.compute_gst(encoder_outputs, style_mel)
        if speaker_embedding is not None:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs.transpose(0, 1), speaker_embedding).transpose(0, 1)
            if self.gst and style_mel is not None:
                self.speaker_embeddings_projected = self.speaker_project_mel(
                        self._concat_speaker_embedding(gst_embedding.transpose(0, 1), speaker_embedding)).squeeze(1)
            else:
                self.speaker_embeddings_projected = self.speaker_project_mel(
                        speaker_embedding).squeeze(1)

        mel_outputs, alignments, stop_tokens = self.decoder.inference_truncated(
            encoder_output, self.speaker_embeddings_projected)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        mel_outputs, mel_outputs_postnet, alignments = self.shape_outputs(
            mel_outputs, mel_outputs_postnet, alignments)
        return mel_outputs, mel_outputs_postnet, alignments, stop_tokens

    def _backward_inference(self, mel_specs, encoder_outputs, mask):
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask,
            self.speaker_embeddings_projected)
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2)
        return decoder_outputs_b, alignments_b

    @staticmethod
    def _add_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)
        outputs = outputs + speaker_embeddings_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, speaker_embeddings):
        speaker_embeddings_ = speaker_embeddings.expand(
            outputs.size(0), outputs.size(1), -1)    
        outputs = torch.cat([outputs, speaker_embeddings_], dim=-1)
        return outputs

