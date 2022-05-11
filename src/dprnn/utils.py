import numpy as np
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import scipy.io.wavfile as wavfile
from itertools import permutations
from apex import amp
import tqdm
import os

EPS = 1e-6

class dataset(data.Dataset):
    def __init__(self,
                mix_lst_path,
                audio_direc,
                mixture_direc,
                batch_size,
                partition='test',
                sampling_rate=16000,
                max_length=10,
                mix_no=2):

        self.minibatch =[]
        self.audio_direc = audio_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.max_length = max_length
        self.C = mix_no

        mix_lst=open(mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))

        mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)

        start = 0
        while True:
            end = min(len(mix_lst), start + batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

    def __getitem__(self, index):
        batch_lst = self.minibatch[index]
        min_length = math.floor(float(batch_lst[-1].split(',')[-1])*self.sampling_rate)

        mixtures=[]
        audios=[]
        for line in batch_lst:
            mixture_path=self.mixture_direc+self.partition+'/'+ line.replace(',','_').replace('/','_')+'.wav'

            _, mixture = wavfile.read(mixture_path)
            mixture = self._audio_norm(mixture[:min_length])
            mixtures.append(mixture)

            target_audio=[]
            line=line.split(',')
            for c in range(self.C):
                audio_path=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
                _, audio = wavfile.read(audio_path)
                target_audio.append(self._audio_norm(audio[:min_length]))
            audios.append(np.asarray(target_audio))
        
        return np.asarray(mixtures)[...,:self.max_length*self.sampling_rate], \
                np.asarray(audios)[...,:self.max_length*self.sampling_rate]

    def __len__(self):
        return len(self.minibatch)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))


class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, partition):
    datasets = dataset(
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                mixture_direc=args.mixture_direc,
                batch_size=args.batch_size,
                partition=partition,
                mix_no=args.C)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler)

    return sampler, generator

@amp.float_function
def cal_si_snr_with_pit(source, estimate_source, reorder_source = False):
    """Calculate SI-SNR with PIT training.
    Args:
        All in torch tensors
        source: [B, C, T], B: batch size, C: no. of speakers, T: sequence length
        estimate_source: [B, C, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()

    # Step 1. Zero-mean norm
    zero_mean_target = source - torch.mean(source, dim=-1, keepdim=True)
    zero_mean_estimate = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C

    # Step 3: Reorder the estimated source
    if reorder_source:
        reorder_estimate_source = _reorder_source(estimate_source, perms, max_snr_idx)
        return max_snr, reorder_estimate_source
    else:
        return max_snr
        
@amp.float_function
def _reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source

if __name__ == '__main__':
    datasets = dataset(
                mix_lst_path='/home/panzexu/datasets/LRS2/audio/2_mix_min/mixture_data_list_2mix_5db.csv',
                audio_direc='/home/panzexu/datasets/LRS2/audio/Audio/',
                visual_direc='/home/panzexu/datasets/LRS2/lip/',
                mixture_direc='/home/panzexu/datasets/LRS2/audio/2_mix_min/',
                batch_size=8,
                partition='train')
    data_loader = data.DataLoader(datasets,
                batch_size = 1,
                shuffle= True,
                num_workers = 1)

    for a_mix, a_tgt in tqdm.tqdm(data_loader):
        pass

    # test = np.ones((2,4,3,4))
    # print(test.shape)
    # print(test[...,:6].shape)
