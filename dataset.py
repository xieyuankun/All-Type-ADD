#!/usr/bin/python3

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import librosa
from torch.utils.data.dataloader import default_collate
import glob
import random
import numpy
import soundfile
from scipy import signal
from RawBoost import process_Rawboost_feature

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


class AudioAugmentor:
    def __init__(self, rir_path='/data2/xyk/RIRS_NOISES', musan_path = '/data2/xyk/musan'):
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = self._load_noiselist(musan_path)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*/*.wav'))

    def _load_noiselist(self, musan_path):
        noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            category = file.split('/')[-3]
            if category not in noiselist:
                noiselist[category] = []
            noiselist[category].append(file)
        return noiselist

    def add_rev(self, audio, audio_length):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :audio_length]

    def add_noise(self, audio, noisecat, audio_length):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = audio_length
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
    
    
    
class asvspoof19(Dataset):
    def __init__(self,  path_to_features, path_to_protocol,  rawboost=False, musanrir=False, audio_length=64600):
        super(asvspoof19, self).__init__()

        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.label = {"spoof": 1, "bonafide": 0}
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.AudioAugmentor = AudioAugmentor()  

        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features, filename + '.flac')
        waveform, sr = torchaudio_load(filepath)

        if self.rawboost:
            waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
            waveform = process_Rawboost_feature(waveform, sr=sr)
        
        waveform = pad_dataset(waveform, self.audio_length)
        
        if self.musanrir:
            audio_length = waveform.size(0)
            waveform = self._apply_augmentation(waveform, audio_length)
        
        label = self.label[label]
        return waveform, filename, label

    def _apply_augmentation(self, waveform, audio_length):
        augtype = random.randint(0, 4)
        
        if augtype == 0:
            return waveform
        elif augtype == 1:
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_rev(waveform.numpy(), audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        elif augtype in [2, 3, 4]:
            noise_type = {2: 'noise', 3: 'speech', 4: 'music'}[augtype]
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_noise(waveform.numpy(), noise_type, audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        
        return waveform

    def collate_fn(self, samples):
        return default_collate(samples)









class codecfakesound(Dataset):
    def __init__(self,  path_to_features,path_to_protocol, rawboost = False ,musanrir = False, audio_length=64600):
        super(codecfakesound, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.label = {"fake": 1, "real": 0}
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.AudioAugmentor = AudioAugmentor() 
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename)
        waveform, sr = torchaudio_load(filepath)
        if self.rawboost:
            waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
            waveform = process_Rawboost_feature(waveform, sr = sr)
        waveform = pad_dataset(waveform, self.audio_length)  
        if self.musanrir:
            audio_length = waveform.size(0)  
            waveform = self._apply_augmentation(waveform, audio_length)
        label = self.label[label]
        return waveform, filename, label
    
    def _apply_augmentation(self, waveform, audio_length):
        augtype = random.randint(0, 4)
        if augtype == 0: 
            return waveform
        elif augtype == 1:  
            waveform = waveform.unsqueeze(dim=0)  
            waveform = self.AudioAugmentor.add_rev(waveform.numpy(), audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        elif augtype in [2, 3, 4]: 
            noise_type = {2: 'noise', 3: 'speech', 4: 'music'}[augtype]
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_noise(waveform.numpy(), noise_type, audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        return waveform
    def collate_fn(self, samples):
        return default_collate(samples)




class singfake(Dataset):
    def __init__(self,  path_to_features,path_to_protocol, rawboost = False ,musanrir = False, audio_length=64600):
        super(singfake, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.label = {"deepfake": 1, "bonafide": 0}
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.AudioAugmentor = AudioAugmentor() 
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        _,_,filename,_,_,label = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename+'.flac')
        waveform, sr = torchaudio_load(filepath)
        if self.rawboost:
            waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
            waveform = process_Rawboost_feature(waveform, sr = sr)
        waveform = pad_dataset(waveform, self.audio_length)  
        if self.musanrir:
            audio_length = waveform.size(0)  
            waveform = self._apply_augmentation(waveform, audio_length)
        label = self.label[label]
        return waveform, filename, label
    
    def _apply_augmentation(self, waveform, audio_length):
        augtype = random.randint(0, 4)
        if augtype == 0: 
            return waveform
        elif augtype == 1:  
            waveform = waveform.unsqueeze(dim=0)  
            waveform = self.AudioAugmentor.add_rev(waveform.numpy(), audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        elif augtype in [2, 3, 4]: 
            noise_type = {2: 'noise', 3: 'speech', 4: 'music'}[augtype]
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_noise(waveform.numpy(), noise_type, audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        return waveform
    def collate_fn(self, samples):
        return default_collate(samples)



class fakemusiccaps(Dataset):
    def __init__(self,  path_to_features,path_to_protocol, rawboost = False ,musanrir = False, audio_length=64600):
        super(fakemusiccaps, self).__init__()
        self.path_to_features = path_to_features
        self.path_to_protocol = path_to_protocol
        self.audio_length = audio_length
        self.label = {"fake": 1, "real": 0}
        self.rawboost = rawboost
        self.musanrir = musanrir
        self.AudioAugmentor = AudioAugmentor() 
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_files = audio_info
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, label, _ = self.all_files[idx]
        filepath = os.path.join(self.path_to_features,filename)
        waveform, sr = torchaudio_load(filepath)
        if self.rawboost:
            waveform = waveform.squeeze(dim=0).detach().cpu().numpy()
            waveform = process_Rawboost_feature(waveform, sr = sr)
        waveform = pad_dataset(waveform, self.audio_length)  
        if self.musanrir:
            audio_length = waveform.size(0)  
            waveform = self._apply_augmentation(waveform, audio_length)
        label = self.label[label]
        return waveform, filename, label
    
    def _apply_augmentation(self, waveform, audio_length):
        augtype = random.randint(0, 4)
        if augtype == 0: 
            return waveform
        elif augtype == 1:  
            waveform = waveform.unsqueeze(dim=0)  
            waveform = self.AudioAugmentor.add_rev(waveform.numpy(), audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        elif augtype in [2, 3, 4]: 
            noise_type = {2: 'noise', 3: 'speech', 4: 'music'}[augtype]
            waveform = waveform.unsqueeze(dim=0)
            waveform = self.AudioAugmentor.add_noise(waveform.numpy(), noise_type, audio_length)
            waveform = torch.tensor(waveform).squeeze(dim=0)
            return waveform
        return waveform
    def collate_fn(self, samples):
        return default_collate(samples)


    
    
if __name__ == "__main__":
    
    dataset = fakemusiccaps(path_to_features="/data7/xyk/fakemusiccaps/eval/")
    print(dataset)    






