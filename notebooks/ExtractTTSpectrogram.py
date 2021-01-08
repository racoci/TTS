#!/usr/bin/env python
# coding: utf-8

# This is a notebook to generate mel-spectrograms from a TTS model to be used for WaveRNN training.

# In[ ]:



import os
import sys
import torch
import importlib
import numpy as np
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
from TTS.tts.datasets.TTSDataset import MyDataset
from TTS.tts.layers.losses import L1LossMasked
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.io import load_checkpoint

from TTS.tts.utils.visual import plot_spectrogram
from TTS.tts.utils.synthesis import synthesis, inv_spectrogram
from TTS.tts.utils.generic_utils import  setup_model
from TTS.utils.io import copy_config_file, load_config
from TTS.tts.utils.text.symbols import make_symbols, symbols, phonemes
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


def format_data(data, speaker_mapping=None):
    if speaker_mapping is None and C.use_speaker_embedding and not C.use_external_speaker_embedding_file:
        speaker_mapping = load_speaker_mapping(OUT_PATH)

    # setup input data
    text_input = data[0]
    text_lengths = data[1]
    speaker_names = data[2]
    linear_input = data[3] if C.model in ["Tacotron"] else None
    mel_input = data[4]
    mel_lengths = data[5]
    stop_targets = data[6]
    avg_text_length = torch.mean(text_lengths.float())
    avg_spec_length = torch.mean(mel_lengths.float())

    if C.use_speaker_embedding:
        if C.use_external_speaker_embedding_file:
            speaker_embeddings = data[8]
            speaker_ids = None
        else:
            speaker_ids = [
                speaker_mapping[speaker_name] for speaker_name in speaker_names
            ]
            speaker_ids = torch.LongTensor(speaker_ids)
            speaker_embeddings = None
    else:
        speaker_embeddings = None
        speaker_ids = None


    # set stop targets view, we predict a single stop token per iteration.
    stop_targets = stop_targets.view(text_input.shape[0],
                                     stop_targets.size(1) // C.r, -1)
    stop_targets = (stop_targets.sum(2) >
                    0.0).unsqueeze(2).float().squeeze(2)

    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda(non_blocking=True)
        text_lengths = text_lengths.cuda(non_blocking=True)
        mel_input = mel_input.cuda(non_blocking=True)
        mel_lengths = mel_lengths.cuda(non_blocking=True)
        linear_input = linear_input.cuda(non_blocking=True) if C.model in ["Tacotron"] else None
        stop_targets = stop_targets.cuda(non_blocking=True)
        if speaker_ids is not None:
            speaker_ids = speaker_ids.cuda(non_blocking=True)
        if speaker_embeddings is not None:
            speaker_embeddings = speaker_embeddings.cuda(non_blocking=True)

    return text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, speaker_ids, speaker_embeddings, avg_text_length, avg_spec_length


def set_filename(wav_path, out_path):
    wav_file = os.path.basename(wav_path)
    file_name = wav_file.split('.')[0]+'.npy'
    os.makedirs(os.path.join(out_path, "quant"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "mel"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "wav_gl"), exist_ok=True)
    wavq_path = os.path.join(out_path, "quant", file_name)
    mel_path = os.path.join(out_path, "mel", file_name)
    wav_path = os.path.join(out_path, "wav_gl", file_name)
    return file_name, wavq_path, mel_path, wav_path


# In[ ]:



OUT_PATH = "../../../hifi-Gan-data/VCTK-Tacotron/"
DATA_PATH = "../../../../datasets/VCTK-Corpus-removed-silence/"
DATASET = "vctk"
METADATA_FILE = None


CONFIG_PATH = "../../checkpoints-Tacotron2/paper-do-zero/vctk-r=2-ddc-TL-my-model-December-10-2020_08+31PM-2868995/config.json"
MODEL_FILE = "../../checkpoints-Tacotron2/paper-do-zero/vctk-r=2-ddc-TL-my-model-December-10-2020_08+31PM-2868995/best_model.pth.tar"
SPEAKER_JSON = '../../checkpoints-Tacotron2/paper-do-zero/vctk-r=2-ddc-TL-my-model-December-10-2020_08+31PM-2868995/speakers.json'

BATCH_SIZE = 32

QUANTIZED_WAV = False
QUANTIZE_BIT = 9
DRY_RUN = False   # if False, does not generate output files, only computes loss and visuals.

use_cuda = torch.cuda.is_available()
print(" > CUDA enabled: ", use_cuda)

C = load_config(CONFIG_PATH)
C.audio['do_trim_silence'] = False  # IMPORTANT!!!!!!!!!!!!!!! disable to a

# C.audio['stats_path'] = '../../checkpoints-Tacotron2/2136433/tts_scale_stats.npy'
ap = AudioProcessor(**C.audio)



# if the vocabulary was passed, replace the default
if 'characters' in C.keys():
    symbols, phonemes = make_symbols(**C.characters)

# load the model
SPEAKER_FILEID = None # if None use the first embedding from speakers.json
if SPEAKER_JSON != '':
    speaker_mapping = json.load(open(SPEAKER_JSON, 'r'))
    num_speakers = len(speaker_mapping)
    if C.use_external_speaker_embedding_file:
        if SPEAKER_FILEID is not None:
            speaker_embedding = speaker_mapping[SPEAKER_FILEID]['embedding']
        else: # if speaker_fileid is not specificated use the first sample in speakers.json
            choise_speaker = list(speaker_mapping.keys())[0]
            print(" Speaker: ",choise_speaker.split('_')[0],'was chosen automatically', "(this speaker seen in training)")
            speaker_embedding = speaker_mapping[choise_speaker]['embedding']
        speaker_embedding_dim = len(speaker_embedding)
    
if 'characters' in C.keys():
    symbols, phonemes = make_symbols(**C.characters)

# load the model
num_chars = len(phonemes) if C.use_phonemes else len(symbols)
model = setup_model(num_chars, num_speakers, C, speaker_embedding_dim)

model, checkpoint =  load_checkpoint(model, MODEL_FILE, use_cuda=use_cuda)
model.eval()
model.decoder.set_r(checkpoint['r'])

# set config r
C.r = checkpoint['r']

if use_cuda:
    model = model.cuda()


# In[ ]:


preprocessor = importlib.import_module('TTS.tts.datasets.preprocess')
preprocessor = getattr(preprocessor, DATASET.lower())
meta_data = preprocessor(DATA_PATH,METADATA_FILE)
dataset = MyDataset(checkpoint['r'], C.text_cleaner, False, ap, meta_data,tp=C.characters if 'characters' in C.keys() else None, use_phonemes=C.use_phonemes,  phoneme_cache_path=C.phoneme_cache_path, enable_eos_bos=C.enable_eos_bos_chars, speaker_mapping=speaker_mapping if C.use_speaker_embedding and C.use_external_speaker_embedding_file else None)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=dataset.collate_fn, shuffle=False, drop_last=False)



# ### Generate model outputs 

# In[ ]:


import pickle

file_idxs = []
metadata = []
losses = []
postnet_losses = []
criterion = L1LossMasked(seq_len_norm=C.seq_len_norm)
with torch.no_grad():
    for data in tqdm(loader):
        item_idx = data[7]
        text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, speaker_ids, speaker_embeddings, avg_text_length, avg_spec_length = format_data(data, speaker_mapping)
        if C.bidirectional_decoder or C.double_decoder_consistency:
            mel_outputs, postnet_outputs, alignments, stop_tokens, _, _ = model.forward(text_input, text_lengths, mel_input, speaker_ids=speaker_ids, speaker_embeddings=speaker_embeddings)
        else:
            mel_outputs, postnet_outputs, alignments, stop_tokens = model.forward(text_input, text_lengths, mel_input, speaker_ids=speaker_ids, speaker_embeddings=speaker_embeddings)
        
        # compute loss
        loss = criterion(mel_outputs, mel_input, mel_lengths)
        loss_postnet = criterion(postnet_outputs, mel_input, mel_lengths)
        losses.append(loss.item())
        postnet_losses.append(loss_postnet.item())

        # compute mel specs from linear spec if model is Tacotron
        if C.model == "Tacotron":
            mel_specs = []
            postnet_outputs = postnet_outputs.data.cpu().numpy()
            for b in range(postnet_outputs.shape[0]):
                postnet_output = postnet_outputs[b]
                mel_specs.append(torch.FloatTensor(ap.out_linear_to_mel(postnet_output.T).T).cuda())
            postnet_outputs = torch.stack(mel_specs)
        elif C.model == "Tacotron2":
            postnet_outputs = postnet_outputs.detach().cpu().numpy()
        alignments = alignments.detach().cpu().numpy()

        if not DRY_RUN:
            for idx in range(text_input.shape[0]):
                wav_file_path = item_idx[idx]

                file_name, wavq_path, mel_path, wav_path = set_filename(wav_file_path, OUT_PATH)
                file_idxs.append(file_name)

                # quantize and save wav
                if QUANTIZED_WAV:
                    wav = ap.load_wav(wav_file_path)
                    wavq = ap.quantize(wav)
                    np.save(wavq_path, wavq)

                # save TTS mel
                mel = postnet_outputs[idx]
                mel_length = mel_lengths[idx]
                mel = mel[:mel_length, :].T
                np.save(mel_path, mel)

                # debug
                # print(mel_path)
                # wav = inv_spectrogram(mel.T, ap, C)
                # ap.save_wav(wav, 'test.wav')
                metadata.append([wav_file_path, mel_path])

    # for wavernn
    if not DRY_RUN:
        pickle.dump(file_idxs, open(OUT_PATH+"/dataset_ids.pkl", "wb"))      
    
    # for pwgan
    with open(os.path.join(OUT_PATH, "metadata.txt"), "w") as f:
        for data in metadata:
            f.write(f"{data[0]}|{data[1]+'.npy'}\n")

    print(np.mean(losses))
    print(np.mean(postnet_losses))


# In[ ]:


# for pwgan
with open(os.path.join(OUT_PATH, "metadata.txt"), "w") as f:
    for data in metadata:
        f.write(f"{data[0]}|{data[1]+'.npy'}\n")


# ### Sanity Check

# In[ ]:


idx = 1
ap.melspectrogram(ap.load_wav(item_idx[idx])).shape


# In[ ]:


import soundfile as sf
wav, sr = sf.read(item_idx[idx])
mel_postnet = postnet_outputs[idx][:mel_lengths[idx], :]
mel_decoder = mel_outputs[idx][:mel_lengths[idx], :].detach().cpu().numpy()
mel_truth = ap.melspectrogram(wav)
print(mel_truth.shape)


# In[ ]:


# plot posnet output
plot_spectrogram(mel_postnet, ap);
print(mel_postnet[:mel_lengths[idx], :].shape)


# In[ ]:


# plot decoder output
plot_spectrogram(mel_decoder, ap);
print(mel_decoder.shape)


# In[ ]:


# plot GT specgrogram
print(mel_truth.shape)
plot_spectrogram(mel_truth.T, ap);


# In[ ]:


# postnet, decoder diff
from matplotlib import pylab as plt
mel_diff = mel_decoder - mel_postnet
plt.figure(figsize=(16, 10))
plt.imshow(abs(mel_diff[:mel_lengths[idx],:]).T,aspect="auto", origin="lower");
plt.colorbar()
plt.tight_layout()


# In[ ]:


# PLOT GT SPECTROGRAM diff
from matplotlib import pylab as plt
mel_diff2 = mel_truth.T - mel_decoder
plt.figure(figsize=(16, 10))
plt.imshow(abs(mel_diff2).T,aspect="auto", origin="lower");
plt.colorbar()
plt.tight_layout()


# In[ ]:


# PLOT GT SPECTROGRAM diff
from matplotlib import pylab as plt
mel = postnet_outputs[idx]
mel_diff2 = mel_truth.T - mel[:mel_truth.shape[1]]
plt.figure(figsize=(16, 10))
plt.imshow(abs(mel_diff2).T,aspect="auto", origin="lower");
plt.colorbar()
plt.tight_layout()

