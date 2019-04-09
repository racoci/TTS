'''
Extract spectrograms and save them to file for training
'''
import os
import sys
import time
import glob
import argparse
import librosa
import importlib
import numpy as np
import tqdm
from utils.generic_utils import load_config, copy_config_file
from utils.audio import AudioProcessor

from multiprocessing import Pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Data folder.')
    parser.add_argument('--cache_path', type=str, help='Cache folder, place to output all the spectrogram files.')
    parser.add_argument(
        '--config', type=str, help='conf.json file for run settings.')
    parser.add_argument(
        "--num_proc", type=int, default=8, help="number of processes.")
    parser.add_argument(
        "--trim_silence",
        type=bool,
        default=False,
        help="trim silence in the voice clip.")
    parser.add_argument("--only_world", type=bool, default=False, help="If True, only worldsceptrogram is extracted.")
    parser.add_argument("--dataset", type=str, help="Target dataset to be processed.")
    parser.add_argument("--val_split", type=int, default=0, help="Number of instances for validation.")
    parser.add_argument("--meta_file", type=str, help="Meta data file to be used for the dataset.")
    parser.add_argument("--process_audio", type=bool, default=False, help="Preprocess audio files.")
    args = parser.parse_args()

    DATA_PATH = args.data_path
    CACHE_PATH = args.cache_path
    CONFIG = load_config(args.config)

    # load the right preprocessor 
    preprocessor = importlib.import_module('datasets.preprocess')
    preprocessor = getattr(preprocessor, args.dataset.lower())
    items = preprocessor(args.data_path, args.meta_file)

    print(" > Input path: ", DATA_PATH)
    print(" > Cache path: ", CACHE_PATH)

    ap = AudioProcessor(**CONFIG.audio)


    def extract_world(item):
        """ Compute spectrograms, length information """
        text = item[0]
        file_path = item[1]
        x = ap.load_wav(file_path, ap.sample_rate)
        file_name = os.path.basename(file_path).replace(".wav", "")
        world_file = file_name + "_world"
        world_path = os.path.join(CACHE_PATH, 'world', world_file)
        world = ap.wav2world(x.astype('float32')).astype('float32')
        np.save(world_path, world, allow_pickle=False)
        world_len = world.shape[1]
        wav_len = x.shape[0]
        output = [text, file_path, world_path+".npy", str(wav_len), str(world_len)]
        if args.process_audio:
            audio_file = file_name + "_audio"
            audio_path = os.path.join(CACHE_PATH, 'audio', audio_file)
            np.save(audio_path, x, allow_pickle=False)
            del output[0]
            output.insert(1, audio_path+".npy")
        return output


    if __name__ == "__main__":
        print(" > Number of files: %i" % (len(items)))
        if not os.path.exists(CACHE_PATH):
            os.makedirs(os.path.join(CACHE_PATH, 'world'))
            if args.process_audio:
                os.makedirs(os.path.join(CACHE_PATH, 'audio'))
            print(" > A new folder created at {}".format(CACHE_PATH))

        # Extract features
        r = []
        if args.num_proc > 1:
            print(" > Using {} processes.".format(args.num_proc))
            with Pool(args.num_proc) as p:
                r = list(
                    tqdm.tqdm(
                        p.imap(extract_world, items),
                        total=len(items)))
                # r = list(p.imap(extract_world, file_names))
        else:
            print(" > Using single process run.")
            for item in items:
                print(" > ", item[1])
                r.append(extract_world(item))

        # Save meta data 
        if args.cache_path is not None:
            file_path = os.path.join(CACHE_PATH, "tts_metadata_val.csv")
            file = open(file_path, "w")
            for line in r[:args.val_split]:
                line = "| ".join(line)
                file.write(line + '\n')
            file.close()

            file_path = os.path.join(CACHE_PATH, "tts_metadata.csv")
            file = open(file_path, "w")
            for line in r[args.val_split:]:
                line = "| ".join(line)
                file.write(line + '\n')
            file.close()
        
        # copy the used config file to output path for sanity
        copy_config_file(args.config, CACHE_PATH)
