#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
from typing import Tuple
import soundfile as sf


torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath,sr=16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]

def pad_dataset(wav, audio_length=64600):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = audio_length
    if waveform_len >= cut:
        waveform = waveform[:cut]
    else:
        # need to pad
        num_repeats = int(cut / waveform_len) + 1
        waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
    
    return waveform


class codecfake(Dataset):
    def __init__(self, path_to_database, path_to_protocol, part='train'):
        super(codecfake, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = self.ptd
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(os.path.join(self.path_to_protocol, 'train.txt'))
        self.label = {"fake": 0, "real": 1}
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)




class ESC50(Dataset):
    def __init__(self, path_to_database, path_to_protocol, part='train'):
        super(ESC50, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = self.ptd
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(os.path.join(self.path_to_protocol, 'train.txt'))
        self.label = {"fake": 0, "real": 1}
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)

        
class CFAD(Dataset):
    def __init__(self, path_to_database, path_to_protocol, part='train'):
        super(CFAD, self).__init__()
        self.ptd = path_to_database
        self.part = part
        self.path_to_audio = os.path.join(self.ptd, self.part)
        self.path_to_protocol = path_to_protocol
        protocol = os.path.join(os.path.join(self.path_to_protocol, self.part + '.txt'))
        self.label = {"fake": 1, "real": 0}
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label,_ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)





        
class asvspoof19(Dataset):
    def __init__(self,  path_to_features, path_to_protocol, audio_length=64600):
        super(asvspoof19, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.label = {"spoof": 1, "bonafide": 0}
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info
    def __len__(self):
        return len(self.all_files)
    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename+'.flac')
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)
        waveform = waveform.squeeze(dim=0)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)

class codecfakesound(Dataset):
    def __init__(self, path_to_audio, path_to_label, audio_length=64600):   
        super(codecfakesound, self).__init__()
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_label
        self.audio_length = audio_length
        
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info
    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)

        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)       
     
class singfake(Dataset):
    def __init__(self, path_to_audio, path_to_label, audio_length=64600):   
        super(singfake, self).__init__()
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_label
        self.audio_length = audio_length
        
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info
    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        _,_,filename,_,_,label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename + '.flac')
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)

        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)           
     
class fakemusiccaps(Dataset):
    def __init__(self, path_to_audio, path_to_label, audio_length=64600):   
        super(fakemusiccaps, self).__init__()
        self.path_to_audio = path_to_audio
        self.path_to_protocol = path_to_label
        self.audio_length = audio_length
        
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info
    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label, _ = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio,filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)

        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)         
     
     
     
class ITW(Dataset):
    def __init__(self, path_to_audio, path_to_label, audio_length=64600):
        super(ITW, self).__init__()
        self.path_to_audio = path_to_audio
        self.path_to_label = path_to_label
        self.audio_length = audio_length
        with open(self.path_to_label, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        _,filename, _, _,label = self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename+'.wav')
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)
                                









class CFAD_codec(Dataset):
    def __init__(self, path_to_features, audio_length=64600, genuine_only=False):
        super(CFAD_codec, self).__init__()
        self.path_to_features = path_to_features
        self.audio_length = audio_length
        self.label = {"fake": "fake", "real": "real"}

        self.real_files = []
        self.fake_files = []

        real_folder = os.path.join(path_to_features,  "real_codec")
        fake_folder = os.path.join(path_to_features,  "fake_codec")
        self._collect_wav_files(real_folder, self.real_files)
        self._collect_wav_files(fake_folder, self.fake_files)

        if genuine_only:
            self.all_files = self.real_files
        else:
            self.all_files = self.real_files + self.fake_files

    def _collect_wav_files(self, folder_path, file_list):

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)
        label = self._get_label(filepath)
        return waveform, os.path.basename(filepath), label

    def _get_label(self, filepath):

        if filepath.startswith(os.path.join(self.path_to_features,  "real_codec")):
            return "real"
        return "fake"

    def collate_fn(self, samples):
        return default_collate(samples)


class FSW_eval(Dataset):
    def __init__(self, path_to_features, audio_length=64600, prefix=None):
        super(FSW_eval, self).__init__()
        self.ptf = os.path.join(path_to_features)
        self.audio_length = audio_length
        self.label = {"fake": 1, "real": 0}
        self.all_files = self._get_all_files(prefix)

    def _get_all_files(self, prefix):
        all_files = []
        for root, _, files in os.walk(self.ptf):
            for file in files:
                if file.endswith('.wav'):
                    if prefix is None or file.startswith(prefix):
                        label_char = file.split('_')[-1][0] 
                        label = 'fake' if label_char == 'F' else 'real'
                        all_files.append((file, label))
        return all_files

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, label = self.all_files[idx]
        filepath = os.path.join(self.ptf, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)
        label = self.label[label]
        return waveform, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)

class any_nolabel(Dataset):
    def __init__(self, path_to_audio,  audio_length=64600):
        super(any_nolabel, self).__init__()
        self.path_to_audio = path_to_audio
        self.audio_length = audio_length
        self.all_info = os.listdir(self.path_to_audio)

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename= self.all_info[idx]
        filepath = os.path.join(self.path_to_audio, filename)
        waveform, sr = torchaudio_load(filepath)
        waveform = pad_dataset(waveform, self.audio_length)
        return waveform, filename, 
    def collate_fn(self, samples):
        return default_collate(samples)


        