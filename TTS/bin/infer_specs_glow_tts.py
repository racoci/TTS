#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
import time
import traceback
import importlib

import torch
from tqdm import tqdm
import numpy as np

from random import randrange
from torch.utils.data import DataLoader
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.tts.datasets.TTSDataset import MyDataset
from TTS.tts.layers.losses import GlowTTSLoss
from TTS.tts.utils.distribute import (DistributedSampler, init_distributed,
                                      reduce_tensor)
from TTS.tts.utils.generic_utils import setup_model, check_config_tts
from TTS.tts.utils.io import save_best_model, save_checkpoint
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.speakers import load_speaker_mapping, get_speakers, save_speaker_mapping
from TTS.tts.utils.synthesis import synthesis, inv_spectrogram
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor
from TTS.utils.console_logger import ConsoleLogger
from TTS.utils.generic_utils import (KeepAverage, count_parameters,
                                     create_experiment_folder, get_git_branch,
                                     remove_experiment_folder, set_init_dict)
from TTS.utils.io import copy_config_file, load_config
from TTS.utils.radam import RAdam
from TTS.utils.tensorboard_logger import TensorboardLogger
from TTS.utils.training import (NoamLR, check_update,
                                setup_torch_training_env)

use_cuda, num_gpus = setup_torch_training_env(True, False)

def parse_speakers(c, args, meta_data_train, OUT_PATH):
    """ Returns number of speakers, speaker embedding shape and speaker mapping"""
    if c.use_speaker_embedding:
        speakers = get_speakers(meta_data_train)
        if args.restore_path:
            if c.use_external_speaker_embedding_file: # if restore checkpoint and use External Embedding file
                prev_out_path = os.path.dirname(args.restore_path)
                speaker_mapping = load_speaker_mapping(c.external_speaker_embedding_file)
                speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]['embedding'])
            elif not c.use_external_speaker_embedding_file: # if restore checkpoint and don't use External Embedding file
                prev_out_path = os.path.dirname(args.restore_path)
                speaker_mapping = load_speaker_mapping(prev_out_path)
                speaker_embedding_dim = None
                assert all([speaker in speaker_mapping
                            for speaker in speakers]), "As of now you, you cannot " \
                                                    "introduce new speakers to " \
                                                    "a previously trained model."
        elif c.use_external_speaker_embedding_file and c.external_speaker_embedding_file: # if start new train using External Embedding file
            speaker_mapping = load_speaker_mapping(c.external_speaker_embedding_file)
            speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]['embedding'])
        elif c.use_external_speaker_embedding_file and not c.external_speaker_embedding_file: # if start new train using External Embedding file and don't pass external embedding file
            raise "use_external_speaker_embedding_file is True, so you need pass a external speaker embedding file, run GE2E-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb or AngularPrototypical-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb notebook in notebooks/ folder"
        else: # if start new train and don't use External Embedding file
            speaker_mapping = {name: i for i, name in enumerate(speakers)}
            speaker_embedding_dim = None
        # save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print("Training with {} speakers: {}".format(len(speakers),
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0
        speaker_embedding_dim = None
        speaker_mapping = None

    return num_speakers, speaker_embedding_dim, speaker_mapping

def setup_loader(ap, r, is_val=False, verbose=False, speaker_mapping=None):
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            r,
            c.text_cleaner,
            compute_linear_spec=False,
            meta_data=meta_data_eval if is_val else meta_data_train,
            ap=ap,
            tp=c.characters if 'characters' in c.keys() else None,
            add_blank=c['add_blank'] if 'add_blank' in c.keys() else False,
            batch_group_size=0 if is_val else c.batch_group_size *
            c.batch_size,
            min_seq_len=c.min_seq_len,
            max_seq_len=c.max_seq_len,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            enable_eos_bos=c.enable_eos_bos_chars,
            verbose=verbose,
            speaker_mapping=speaker_mapping if c.use_speaker_embedding and c.use_external_speaker_embedding_file else None)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader


def format_data(data):
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)

    # setup input data
    text_input = data[0]
    text_lengths = data[1]
    speaker_names = data[2]
    mel_input = data[4].permute(0, 2, 1)  # B x D x T
    mel_lengths = data[5]
    attn_mask = data[9]
    wav_files_names = data[7] #[10]

    avg_text_length = torch.mean(text_lengths.float())
    avg_spec_length = torch.mean(mel_lengths.float())

    if c.use_speaker_embedding:
        if c.use_external_speaker_embedding_file:
            speaker_ids = data[8]
        else:
            speaker_ids = [
                speaker_mapping[speaker_name] for speaker_name in speaker_names
            ]
            speaker_ids = torch.LongTensor(speaker_ids)
    else:
        speaker_ids = None

    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda(non_blocking=True)
        text_lengths = text_lengths.cuda(non_blocking=True)
        mel_input = mel_input.cuda(non_blocking=True)
        mel_lengths = mel_lengths.cuda(non_blocking=True)
        if speaker_ids is not None:
            speaker_ids = speaker_ids.cuda(non_blocking=True)
        if attn_mask is not None:
            attn_mask = attn_mask.cuda(non_blocking=True)

    return text_input, text_lengths, mel_input, mel_lengths, speaker_ids,\
         avg_text_length, avg_spec_length, attn_mask, wav_files_names


def data_depended_init(model, ap, speaker_mapping=None):
    """Data depended initialization for activation normalization."""
    if hasattr(model, 'module'):
        for f in model.module.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)
    else:
        for f in model.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)

    data_loader = setup_loader(ap, 1, is_val=False, speaker_mapping=speaker_mapping)
    model.train()
    print(" > Data depended initialization ... ")
    with torch.no_grad():
        for _, data in enumerate(data_loader):

            # format data
            text_input, text_lengths, mel_input, mel_lengths, speaker_ids,\
                _, _, attn_mask, _ = format_data(data)

            # forward pass model
            _ = model.forward(
                text_input, text_lengths, mel_input, mel_lengths, attn_mask, g=speaker_ids)
            break

    if hasattr(model, 'module'):
        for f in model.module.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(False)
    else:
        for f in model.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(False)
    return model



def infer(model, data_loader, speaker_mapping=None, dir_out=None, metada_name='metadata.txt', ap=None, c=None, debug=False):
    if dir_out is not None:
        os.makedirs(dir_out, exist_ok=True)

    model.eval()
    epoch_time = 0
    keep_avg = KeepAverage()
    if use_cuda:
        batch_n_iter = int(
            len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    c_logger.print_train_start()

    metadata = []
    for num_iter, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        start_time = time.time()

        # format data
        text_input, text_lengths, mel_input, mel_lengths, speaker_ids,\
            avg_text_length, avg_spec_length, attn_mask, wav_file_names = format_data(data)

        loader_time = time.time() - end_time

        # forward pass model
        y, logdet, y_mean, y_log_scale, alignments, o_dur_log, o_total_dur = model.inference_with_lenghts(text_input, text_lengths, mel_input, mel_lengths, attn_mask, g=speaker_ids)
        # z, logdet, y_mean, y_log_scale, alignments, o_dur_log, o_total_dur = model.forward(text_input, text_lengths, mel_input, mel_lengths, attn_mask, g=speaker_ids)

        for idx in range(y.shape[0]):
            mel = y[idx].T
            mel_length = mel_lengths[idx]
            mel = mel[:mel_length, :].detach().cpu().numpy()

            if dir_out is not None:
                
                file_name, _ = os.path.splitext(os.path.basename(wav_file_names[idx]))
                metadata.append(file_name+'| ')
                print(dir_out)
                print(file_name)
                print("salvando em ", os.path.join(dir_out, file_name+'.npy'))
                np.save(os.path.join(dir_out, file_name+'.npy'), mel)
                
                if debug:
                    wav = inv_spectrogram(mel, ap, c)
                    ap.save_wav(wav, os.path.join(dir_out, file_name+'.wav'))
                    mel_i = mel_input[idx].T
                    expected_mel = mel_i[:mel_length, :].detach().cpu().numpy()
                    wav_exp = inv_spectrogram(expected_mel, ap, c)
                    wav_exp_unpaded = inv_spectrogram(mel_i.detach().cpu().numpy(), ap, c)
                    ap.save_wav(wav_exp, os.path.join(dir_out, file_name+'_exp.wav'))
                    ap.save_wav(wav_exp_unpaded, os.path.join(dir_out, file_name+'_exp_unpad.wav'))
    if dir_out is not None:
        with open(os.path.join(dir_out, metada_name), "w") as f:
            f.write("\n".join(metadata))

# FIXME: move args definition/parsing inside of main?
def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data_train, meta_data_eval, symbols, phonemes
    # Audio processor
    ap = AudioProcessor(**c.audio)
    if 'characters' in c.keys():
        symbols, phonemes = make_symbols(**c.characters)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    # load data instances
    meta_data = []
    for dataset in c.datasets:
        name = dataset['name']
        root_path = dataset['path']
        meta_file_train = dataset['meta_file_train']
        meta_file_val = dataset['meta_file_val']
        preprocessor = importlib.import_module('TTS.tts.datasets.preprocess')
        preprocessor = getattr(preprocessor, name.lower())
        meta_data_eval = []
        meta_data_train = preprocessor(root_path, meta_file_train)
        if meta_file_val is not None:
            meta_data_eval = preprocessor(root_path, meta_file_val)
        
        meta_data += meta_data_train
        meta_data += meta_data_eval

    meta_data_train = meta_data

    # parse speakers
    num_speakers, speaker_embedding_dim, speaker_mapping = parse_speakers(c, args, meta_data_train, OUT_PATH)

    # setup model
    model = setup_model(num_chars, num_speakers, c, speaker_embedding_dim=speaker_embedding_dim)
    optimizer = RAdam(model.parameters(), lr=c.lr, weight_decay=0, betas=(0.9, 0.98), eps=1e-9)
    criterion = GlowTTSLoss()

    if c.apex_amp_level:
        # pylint: disable=import-outside-toplevel
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level=c.apex_amp_level)
    else:
        amp = None

    if args.restore_path:
        checkpoint = torch.load(args.restore_path, map_location='cpu')
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except: #pylint: disable=bare-except
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint['model'], c)
            model.load_state_dict(model_dict)
            del model_dict

        if amp and 'amp' in checkpoint:
            amp.load_state_dict(checkpoint['amp'])

        for group in optimizer.param_groups:
            group['initial_lr'] = c.lr
        print(" > Model restored from step %d" % checkpoint['step'],
              flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model.cuda()
        criterion.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model = DDP(model)

    if c.noam_schedule:
        scheduler = NoamLR(optimizer,
                           warmup_steps=c.warmup_steps,
                           last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    global_step = args.restore_step

    # model = data_depended_init(model, ap, speaker_mapping)

    data_loader_train = setup_loader(ap, 1, is_val=False,
                               verbose=True, speaker_mapping=speaker_mapping)
                            
    infer(model, data_loader_train, speaker_mapping, dir_out=args.outpu_mel_spec, metada_name="metada_train.txt", ap=ap, c=c)
    #infer(model, data_loader_eval, speaker_mapping, dir_out=args.outpu_mel_spec, metada_name="metada_eval.txt", ap=ap, c=c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue_path',
        type=str,
        help='Training output folder to continue training. Use to continue a training. If it is used, "config_path" is ignored.',
        default='',
        required='--config_path' not in sys.argv)
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
        required='--continue_path' not in sys.argv
    )
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Do not verify commit integrity to run training.')

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument('--group_id',
                        type=str,
                        default="",
                        help='DISTRIBUTED: process group id.')

    parser.add_argument(
        '--outpu_mel_spec',
        type=str,
        help='Path to save mel specs',
        default=None)
    args = parser.parse_args()

    if args.continue_path != '':
        args.output_path = args.continue_path
        args.config_path = os.path.join(args.continue_path, 'config.json')
        list_of_files = glob.glob(args.continue_path + "/*.pth.tar") # * means all if need specific format then *.csv
        latest_model_file = max(list_of_files, key=os.path.getctime)
        args.restore_path = latest_model_file
        print(f" > Training continues for {args.restore_path}")

    # setup output paths and read configs
    c = load_config(args.config_path)
    
    # set batch to 1 for dont use padding
    # c.batch_size = 1

    check_config_tts(c)
    _ = os.path.dirname(os.path.realpath(__file__))

    if c.apex_amp_level:
        print("   >  apex AMP level: ", c.apex_amp_level)

    OUT_PATH = args.continue_path
    if args.continue_path == '':
        OUT_PATH = create_experiment_folder(c.output_path, c.run_name, args.debug)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    c_logger = ConsoleLogger()
    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
