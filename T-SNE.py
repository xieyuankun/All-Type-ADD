import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import torch
import tqdm
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



def extract_feats_and_labels(model, testDataLoader, args, task_type):
    all_feats = []
    all_labels = []
    all_filenames = []
    all_task_types = []  
    
    class_counter = {'real': 0, 'fake': 0}
    max_samples_per_class = 1000  
    
    with torch.no_grad():
        for data_slice in tqdm(testDataLoader):
            waveform, filename, labels = data_slice[0], data_slice[1], data_slice[2]
            
            feats, w2v2_outputs = model(waveform)
            
            scores = F.softmax(w2v2_outputs, dim=1)[:, 0].detach().cpu().numpy()
            
            for fn, score, label in zip(filename, scores, labels):
                if (label == 1 or label == 'fake' or label == 'deepfake') and class_counter['fake'] < max_samples_per_class:
                    all_feats.append(feats.cpu().numpy())
                    all_labels.append(1)  
                    all_filenames.append(fn.strip())
                    all_task_types.append(task_type)  
                    class_counter['fake'] += 1
                elif (label == 0 or label == 'real' or label == 'bonafide') and class_counter['real'] < max_samples_per_class:
                        all_feats.append(feats.cpu().numpy())
                        all_labels.append(0)  
                        all_filenames.append(fn.strip())
                        all_task_types.append(task_type)  
                        class_counter['real'] += 1
                print(class_counter['fake'], class_counter['real'])
            if class_counter['real'] >= max_samples_per_class and class_counter['fake'] >= max_samples_per_class:
                break

    all_feats = np.vstack(all_feats)  
    all_labels = np.array(all_labels)
    all_task_types = np.array(all_task_types)  

    print(f"Real samples: {sum(all_labels == 0)}, Fake samples: {sum(all_labels == 1)}")
    
    return all_feats, all_labels, all_task_types

def plot_tsne(all_feats, all_labels, all_task_types):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_feats = tsne.fit_transform(all_feats)
    
    plt.figure(figsize=(10, 8))
    

    task_colors = {'speech': 'blue', 'sound': 'green', 'singing': 'orange', 'music': 'purple'}
    label_markers = {0: 'x', 1: 'o'}  
    
    for task_type in np.unique(all_task_types):

        task_mask = (all_task_types == task_type)
        for label in [0, 1]:
            label_mask = (all_labels == label)
            mask = task_mask & label_mask  
            
            plt.scatter(reduced_feats[mask, 0], reduced_feats[mask, 1],
                        c=task_colors[task_type], label=f'{task_type} {"real" if label == 0 else "fake"}',
                        marker=label_markers[label], alpha=0.7)
    plt.xticks([])
    plt.yticks([])
    # plt.legend(fontsize=16, loc='upper right', framealpha=0.3)
    # plt.title("T-SNE Visualization of Features by Task Type")
    plt.savefig("figure/TSNE_ft.png", dpi=300)
    plt.show()

def test_and_plot_tsne(model, args):

    all_feats = []
    all_labels = []
    all_task_types = []
    
    test_set = eval_dataset.asvspoof19(path_to_features=args.asvspoof19_eval_audio,
                                        path_to_protocol=args.asvspoof19_eval_label, audio_length=args.audio_len)
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    feats, labels, task_types = extract_feats_and_labels(model, testDataLoader, args, 'speech')
    all_feats.append(feats)
    all_labels.append(labels)
    all_task_types.append(task_types)

    test_set = eval_dataset.codecfakesound(path_to_audio=args.fakesound_audio, 
                                            path_to_label=args.fakesound_eval_label, audio_length=args.audio_len)
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    feats, labels, task_types = extract_feats_and_labels(model, testDataLoader, args, 'sound')
    all_feats.append(feats)
    all_labels.append(labels)
    all_task_types.append(task_types)
    
    test_set = eval_dataset.singfake(path_to_audio=args.singfake_eval_audio, 
                                        path_to_label=args.singfake_eval_label, audio_length=args.audio_len)
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    feats, labels, task_types = extract_feats_and_labels(model, testDataLoader, args, 'singing')
    all_feats.append(feats)
    all_labels.append(labels)
    all_task_types.append(task_types)

    test_set = eval_dataset.fakemusiccaps(path_to_audio=args.fakemusiccaps_audio, 
                                            path_to_label=args.fakemusiccaps_eval_label, audio_length=args.audio_len)
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    feats, labels, task_types = extract_feats_and_labels(model, testDataLoader, args, 'music')
    all_feats.append(feats)
    all_labels.append(labels)
    all_task_types.append(task_types)
    
    all_feats = np.vstack(all_feats)
    all_labels = np.concatenate(all_labels)
    all_task_types = np.concatenate(all_task_types)
    
    plot_tsne(all_feats, all_labels, all_task_types)


if __name__ == "__main__":
    args = init()
    
    # load model
    ckpt_path = os.path.join(args.model_path, "anti-spoofing_feat_model.pt")
    checkpoint = torch.load(ckpt_path)
    
    if args.model == 'aasist':
        feat_model = Rawaasist().cuda()
    if args.model == 'specresnet':
        feat_model = ResNet18ForAudio().cuda()
    if args.model == 'fr-w2v2aasist':
        feat_model = XLSRAASIST(model_dir=args.xlsr).cuda()
    if args.model == 'fr-wavlmaasist':
        feat_model = WAVLMAASIST(model_dir=args.wavlm).cuda()
    if args.model == 'fr-mertaasist':
        feat_model = MERTAASIST(model_dir=args.mert).cuda()
    if args.model == 'ft-w2v2aasist':
        feat_model = XLSRAASIST(model_dir=args.xlsr, freeze=False).cuda()
    if args.model == 'ft-wavlmaasist':
        feat_model = WAVLMAASIST(model_dir=args.wavlm, freeze=False).cuda()
    if args.model == 'ft-mertaasist':
        feat_model = MERTAASIST(model_dir=args.mert, freeze=False).cuda()
    if args.model == 'pt-w2v2aasist':
        feat_model = PTW2V2AASIST(model_dir=args.xlsr, prompt_dim=args.prompt_dim,
                                  num_prompt_tokens=args.num_prompt_tokens, dropout=args.pt_dropout).cuda()
    if args.model == "wpt-w2v2aasist":
        feat_model = WPTW2V2AASIST(model_dir=args.xlsr, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens=args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens,
                                   dropout=args.pt_dropout).cuda()
    if args.model == 'shpt-w2v2aasist':
        feat_model = PTW2V2AASIST_shallow(model_dir=args.xlsr, prompt_dim=args.prompt_dim,
                                          num_prompt_tokens=args.num_prompt_tokens, dropout=args.pt_dropout).cuda()
    if args.model == 'pt-wavlmaasist':
        feat_model = PTWAVLMAASIST(model_dir=args.wavlm, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens=args.num_prompt_tokens, dropout=args.pt_dropout).cuda()
    if args.model == "wpt-wavlmaasist":
        feat_model = WPTWAVLMAASIST(model_dir=args.wavlm, prompt_dim=args.prompt_dim,
                                    num_prompt_tokens=args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens,
                                    dropout=args.pt_dropout).cuda()
    if args.model == 'pt-mertaasist':
        feat_model = PTMERTAASIST(model_dir=args.mert, prompt_dim=args.prompt_dim,
                                  num_prompt_tokens=args.num_prompt_tokens, dropout=args.pt_dropout).cuda()
    if args.model == "wpt-mertaasist":
        feat_model = WPTMERTAASIST(model_dir=args.mert, prompt_dim=args.prompt_dim,
                                   num_prompt_tokens=args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens,
                                   dropout=args.pt_dropout).cuda()

    feat_model.load_state_dict(checkpoint)
    feat_model.eval()

    # Run the T-SNE visualization
    test_and_plot_tsne(feat_model, args)
