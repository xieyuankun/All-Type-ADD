from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm, trange
import eval_dataset
import numpy as np
from backbone.rawaasist import *
torch.multiprocessing.set_start_method('spawn', force=True)
import config
import json
import argparse

def init():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, required=True, help="Path to the save model file")
    parser.add_argument("--gpu", type=str, help="GPU index", default="4")
    parser.add_argument("--task", type=str, help="Task type", default="speech", 
                        choices=['speech', 'sound', 'singing', 'music'])
    parser.add_argument("--batch_size", type=int, default=None)
    
    temp_args, _ = parser.parse_known_args()
    
    json_path = os.path.join(temp_args.model_path, 'args.json')
    with open(json_path, 'r') as f:
        json_args = json.load(f)
    
    for key, value in json_args.items():
            if key not in vars(temp_args):
                if isinstance(value, bool):
                    parser.add_argument(f'--{key}', 
                        action='store_true' if value else 'store_false',
                        default=value)
                else:
                    parser.add_argument(f'--{key}', 
                        type=type(value), 
                        default=value)   
    args = parser.parse_args()
    

    if args.batch_size is None:
        args.batch_size = json_args.get('batch_size', None)
    print(args.gpu)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args



def test_on_speech(model, args):
    result_dir = os.path.join(args.model_path,'result')
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, 'speech.txt')
    
    test_set = eval_dataset.asvspoof19(path_to_features= args.asvspoof19_eval_audio,
                                       path_to_protocol= args.asvspoof19_eval_label, audio_length=args.audio_len)
                                 
    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    with torch.no_grad():
        with open(file_path, 'w') as cm_score_file:
            for idx, data_slice in enumerate(tqdm(testDataLoader)):
                waveform, filename,labels = data_slice[0],data_slice[1],data_slice[2] 

                feats, w2v2_outputs = model(waveform)

                scores = F.softmax(w2v2_outputs, dim=1)[:, 0].detach().cpu().numpy()
                for fn, score, label in zip(filename, scores, labels):
                    audio_fn = fn.strip().split('.')[0]
                    label_str = "fake" if label == 1 else "real"
                    cm_score_file.write(f'{audio_fn} {score} {label_str}\n')
                

def test_on_sound(model, args):
    result_dir = os.path.join(args.model_path,'result')
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, 'sound.txt')
    
    test_set = eval_dataset.codecfakesound(path_to_audio=args.fakesound_audio, 
                                             path_to_label=args.fakesound_eval_label, audio_length=args.audio_len)

    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    with torch.no_grad():
        with open(file_path, 'w') as cm_score_file:
            for idx, data_slice in enumerate(tqdm(testDataLoader)):
                waveform, filename,labels = data_slice[0],data_slice[1],data_slice[2] 
                feats, w2v2_outputs = model(waveform)

                scores = F.softmax(w2v2_outputs, dim=1)[:, 0].detach().cpu().numpy()
                for fn, score, label in zip(filename, scores, labels):
                    audio_fn = fn.strip().split('.')[0]
                    label_str = "fake" if label == 'fake' else "real"
                    cm_score_file.write(f'{audio_fn} {score} {label_str}\n')


def test_on_singing(model, args):
    
    result_dir = os.path.join(args.model_path,'result')
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, 'singing.txt')
    
    test_set = eval_dataset.singfake(path_to_audio=args.singfake_eval_audio, 
                                             path_to_label=args.singfake_eval_label, audio_length=args.audio_len)

    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    with torch.no_grad():
        with open(file_path, 'w') as cm_score_file:
            for idx, data_slice in enumerate(tqdm(testDataLoader)):
                waveform, filename,labels = data_slice[0],data_slice[1],data_slice[2] 
                feats, w2v2_outputs = model(waveform)

                scores = F.softmax(w2v2_outputs, dim=1)[:, 0].detach().cpu().numpy()
                for fn, score, label in zip(filename, scores, labels):
                    audio_fn = fn.strip().split('.')[0]
                    label_str = "fake" if label == 'deepfake' else "real"
                    cm_score_file.write(f'{audio_fn} {score} {label_str}\n')


def test_on_music(model, args):
        
    result_dir = os.path.join(args.model_path,'result')
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, 'music.txt')
    
    test_set = eval_dataset.fakemusiccaps(path_to_audio=args.fakemusiccaps_audio, 
                                             path_to_label=args.fakemusiccaps_eval_label, audio_length=args.audio_len)

    testDataLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    with torch.no_grad():
        with open(file_path, 'w') as cm_score_file:
            for idx, data_slice in enumerate(tqdm(testDataLoader)):
                waveform, filename,labels = data_slice[0],data_slice[1],data_slice[2] 
                feats, w2v2_outputs = model(waveform)

                scores = F.softmax(w2v2_outputs, dim=1)[:, 0].detach().cpu().numpy()
                for fn, score, label in zip(filename, scores, labels):
                    audio_fn = fn.strip().split('.')[0]
                    label_str = "fake" if label == 'fake' else "real"
                    cm_score_file.write(f'{audio_fn} {score} {label_str}\n')

                
                       
                
if __name__ == "__main__":
    args = init()
    # load model
    ckpt_path = os.path.join(args.model_path, "anti-spoofing_feat_model.pt")
    checkpoint = torch.load(ckpt_path)
    print(args.model)
    if args.model == 'aasist':
        feat_model = Rawaasist().cuda()
    if args.model == 'specresnet':
        feat_model = ResNet18ForAudio().cuda()  
    if args.model == 'fr-w2v2aasist':   #❄
        feat_model = XLSRAASIST(model_dir= args.xlsr).cuda()
    if args.model == 'fr-wavlmaasist':   #❄
        feat_model = WAVLMAASIST(model_dir= args.wavlm).cuda()
    if args.model == 'fr-mertaasist':   #❄
        feat_model = MERTAASIST(model_dir= args.mert).cuda()
    if args.model == 'ft-w2v2aasist':   #🔥
        feat_model = XLSRAASIST(model_dir= args.xlsr, freeze = False).cuda()
    if args.model == 'ft-wavlmaasist':   #🔥
        feat_model = WAVLMAASIST(model_dir= args.wavlm, freeze = False).cuda()
    if args.model == 'ft-mertaasist':   #🔥
        feat_model = MERTAASIST(model_dir= args.mert, freeze = False).cuda()
    if args.model == 'pt-w2v2aasist':   #⭐⭐⭐
        feat_model = PTW2V2AASIST(model_dir= args.xlsr, prompt_dim=args.prompt_dim, 
                                  num_prompt_tokens = args.num_prompt_tokens, dropout= args.pt_dropout).cuda()
    if args.model == "wpt-w2v2aasist":  #⭐⭐⭐⭐⭐
        feat_model = WPTW2V2AASIST(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens, 
                                   dropout= args.pt_dropout).cuda()


# ABLATION      
    if args.model == 'pt-wavlmaasist':  
        feat_model = PTWAVLMAASIST(model_dir= args.wavlm, prompt_dim=args.prompt_dim, 
                                  num_prompt_tokens = args.num_prompt_tokens, dropout= args.pt_dropout).cuda()
    if args.model == "wpt-wavlmaasist": 
        feat_model = WPTWAVLMAASIST(model_dir= args.wavlm, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens, 
                                   dropout= args.pt_dropout).cuda()        
    if args.model == 'pt-mertaasist':  
        feat_model = PTMERTAASIST(model_dir= args.mert, prompt_dim=args.prompt_dim, 
                                  num_prompt_tokens = args.num_prompt_tokens, dropout= args.pt_dropout).cuda()
    if args.model == "wpt-mertaasist":  
        feat_model = WPTMERTAASIST(model_dir= args.mert, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens = args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens, 
                                   dropout= args.pt_dropout).cuda()   
        
        
    feat_model.load_state_dict(checkpoint)
    feat_model.eval()
    
    # test on different dataset
    if args.task == 'speech':
        test_on_speech(feat_model,args)
    elif args.task == 'sound':
        test_on_sound(feat_model,args)
    elif args.task == 'singing':
        test_on_singing(feat_model,args)
    elif args.task == 'music':
        test_on_music(feat_model,args)
    else:
        print('task not supported')
        exit(0)
        