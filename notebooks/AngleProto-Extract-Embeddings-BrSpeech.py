#!/usr/bin/env python
# coding: utf-8

# This is a noteboook used to generate the speaker embeddings with the AngleProto speaker encoder model for multi-speaker training.
# 
# Before running this script please DON'T FORGET: 
# - to set file paths.
# - to download related model files from TTS.
# - download or clone related repos, linked below.
# - setup the repositories. ```python setup.py install```
# - to checkout right commit versions (given next to the model) of TTS.
# - to set the right paths in the cell below.
# 
# Repository:
# - TTS: https://github.com/mozilla/TTS

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import os
import importlib
import random
import librosa
import torch

import numpy as np
from tqdm import tqdm
from TTS.tts.utils.speakers import save_speaker_mapping, load_speaker_mapping
from TTS.speaker_encoder.model import SpeakerEncoder
# you may need to change this depending on your system
os.environ['CUDA_VISIBLE_DEVICES']='0'


from TTS.tts.utils.speakers import save_speaker_mapping, load_speaker_mapping
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config


# You should also adjust all the path constants to point at the relevant locations for you locally

# In[ ]:


MODEL_RUN_PATH = "/mnt/edresson/voice-cloning/glowtts/checkpoint-SpeakerEncoder/"
MODEL_PATH = MODEL_RUN_PATH + "320k.pth.tar"
CONFIG_PATH = MODEL_RUN_PATH + "config.json"


DATASETS_NAME = ['brspeech', 'brspeech'] # list the datasets
DATASETS_PATH = ['/mnt/fred/BRSpeech-Datasets/TTS/BRSpeech-TTS-3.0-beta7/', '/mnt/fred/BRSpeech-Datasets/TTS/BRSpeech-TTS-3.0-beta7/']

DATASETS_METAFILE = ['metadata_train_Mozilla_TTS.csv','metadata_test_Mozilla_TTS.csv']

#DATASETS_NAME = ['brspeech'] # list the datasets
#DATASETS_PATH = ['../../../../datasets/BRSpeech-3-Speakers-Paper/TTS-Portuguese_Corpus/']

#DATASETS_METAFILE = ['test_TTS-Portuguese_Corpus_metadata.csv']
USE_CUDA = True


# In[ ]:


#Preprocess dataset
meta_data = []
for i in range(len(DATASETS_NAME)):
    preprocessor = importlib.import_module('TTS.tts.datasets.preprocess')
    preprocessor = getattr(preprocessor, DATASETS_NAME[i].lower())
    meta_data += preprocessor(DATASETS_PATH[i],DATASETS_METAFILE[i])
      
meta_data= list(meta_data)


# In[ ]:
c = load_config(CONFIG_PATH)
ap = AudioProcessor(**c['audio'])


model = SpeakerEncoder(**c.model)
model.load_state_dict(torch.load(MODEL_PATH)['model'])
model.eval()
if USE_CUDA:
    model.cuda()

embeddings_dict = {}
len_meta_data = len(meta_data)

for i in tqdm(range(len_meta_data)):
    _, wav_file, speaker_id = meta_data[i]
    wav_file_name = os.path.basename(wav_file)
    mel_spec = ap.melspectrogram(ap.load_wav(wav_file, sr=ap.sample_rate)).T
    mel_spec = torch.FloatTensor(mel_spec[None, :, :])
    if USE_CUDA:
        mel_spec = mel_spec.cuda()
    embedd = model.compute_embedding(mel_spec).cpu().detach().numpy().reshape(-1)
    if wav_file_name in embeddings_dict.keys():
        raise Exception('Problem Repeat Keys')
    embeddings_dict[wav_file_name] = [embedd, speaker_id]

# In[ ]:


# create and export speakers.json
speaker_mapping = {sample: {'name': embeddings_dict[sample][1], 'embedding':embeddings_dict[sample][0].reshape(-1).tolist()} for i, sample in enumerate(embeddings_dict.keys())}
save_speaker_mapping(MODEL_RUN_PATH, speaker_mapping)


# In[ ]:


#test load integrity
speaker_mapping_load = load_speaker_mapping(MODEL_RUN_PATH)
assert speaker_mapping == speaker_mapping_load
print("The file speakers.json has been exported to ",MODEL_RUN_PATH, ' with ', len(embeddings_dict.keys()), ' speakers')


