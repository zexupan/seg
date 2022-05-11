import torch
import lmdb, pyarrow
import scipy.io.wavfile as wavfile
import numpy as np
import math
import torch.nn as nn
import torch.utils.data as data
import tqdm, os
import argparse

EPS = 1e-6

MAX_INT16 = np.iinfo(np.int16).max

def write_wav(fname, samps, sampling_rate=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    samps = np.divide(samps, np.max(np.abs(samps)))

    # same as MATLAB and kaldi
    if normalize:
        samps = samps * MAX_INT16
        samps = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    if not os.path.exists(fname):
        wavfile.write(fname, sampling_rate, samps)

def write_npy(fname, samps):
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    if not os.path.exists(fname):
        np.save(fname, samps)

class SpeechMotionDataset(data.Dataset):
	def __init__(self, preloaded_dir):
		self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
		with self.lmdb_env.begin() as txn:
			self.n_samples = txn.stat()['entries']

	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		with self.lmdb_env.begin(write=False) as txn:
			key = '{:010}'.format(idx).encode('ascii')
			sample = txn.get(key)

			sample = pyarrow.deserialize(sample)
			word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
		return word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info

def save_data(audio_cache, pose_seq_cache, name, audio_save_path, pose_save_path):
    # print(audio_cache.shape[0])
    # print(pose_seq_cache.shape[0])
    assert int(audio_cache.shape[0]/pose_seq_cache.shape[0]) == int(16000/15)

    save_name = name[0] + '/' + str(name[1]) + '_'+ str(name[2])
    # print(save_name)
    write_wav(audio_save_path + save_name + '.wav', audio_cache)
    write_npy(pose_save_path + save_name + '.npy', pose_seq_cache)
    # print(name[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pose dataset')
    parser.add_argument('--data_direc', type=str)
    parser.add_argument('--audio_data_direc', type=str)
    parser.add_argument('--pose_data_direc', type=str)
    args = parser.parse_args()

    for partition in ["test/", "val/", "train/"]:
        
        preloaded_dir = args.data_direc + partition # data load directory  
        audio_save_path = args.audio_data_direc + partition # speech data save directory
        pose_save_path = args.pose_data_direc  + partition 

        datasets = SpeechMotionDataset(preloaded_dir)
        data_loader = data.DataLoader(datasets,
                    batch_size = 1,
                    shuffle= False,
                    num_workers = 1)


        vid_dic={}

        pre_vid = None

        for word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info in tqdm.tqdm(data_loader):

            vid = aux_info['vid'][0]
            start_frame_no = aux_info['start_frame_no'].squeeze().numpy()
            end_frame_no = aux_info['end_frame_no'].squeeze().numpy()

            audio = audio.squeeze().numpy()
            pose_seq = pose_seq.squeeze().numpy()

            if vid != pre_vid:
                if pre_vid !=None:
                    name = (pre_vid, ini_start_frame_no, pre_end_frame_no)
                    save_data(audio_cache, pose_seq_cache, name, audio_save_path, pose_save_path)
                pre_vid = vid
                ini_start_frame_no = start_frame_no
                pre_end_frame_no = end_frame_no
                audio_cache = audio
                pose_seq_cache = pose_seq
            else:
                if (end_frame_no - pre_end_frame_no) > 42:
                    name = (pre_vid, ini_start_frame_no, pre_end_frame_no)
                    save_data(audio_cache, pose_seq_cache, name, audio_save_path, pose_save_path)

                    ini_start_frame_no = start_frame_no
                    pre_end_frame_no = end_frame_no
                    audio_cache = audio
                    pose_seq_cache = pose_seq
                else:
                    cache = end_frame_no - pre_end_frame_no
                    pre_end_frame_no = end_frame_no
                    pose_seq_cache = np.append(pose_seq_cache, pose_seq[-cache:], 0)
                    audio_cache = np.append(audio_cache[:int((pose_seq_cache.shape[0]-42)/15*16000)], audio, 0)
                    


