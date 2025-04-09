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

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns

def visual_dataset(model, args):
    visual_dir = os.path.join(args.model_path, 'figure')    
    os.makedirs(visual_dir, exist_ok=True)
    speech_test_set = eval_dataset.asvspoof19(
        path_to_features=args.asvspoof19_eval_audio,
        path_to_protocol=args.asvspoof19_eval_label,
        audio_length=args.audio_len
    )
    
    sound_test_set = eval_dataset.codecfakesound(path_to_audio=args.fakesound_audio, 
                                             path_to_label=args.fakesound_eval_label, audio_length=args.audio_len)
        
    sing_test_set = eval_dataset.singfake(path_to_audio=args.singfake_eval_audio, 
                                             path_to_label=args.singfake_eval_label, audio_length=args.audio_len)
    
    music_test_set = eval_dataset.fakemusiccaps(path_to_audio=args.fakemusiccaps_audio, 
                                             path_to_label=args.fakemusiccaps_eval_label, audio_length=args.audio_len)
    
    testDataLoader = DataLoader(sing_test_set, batch_size=10, shuffle=False, num_workers=0)
    # 
    if model is None:
        raise ValueError("Model is not initialized.")

    with torch.no_grad():
        for idx, data_slice in enumerate(tqdm(testDataLoader)):
            waveform, filename, labels = data_slice[0], data_slice[1], data_slice[2] 
            feats, w2v2_outputs, attweight = model(waveform)     #(attweight 24 tuple, each tuple [batch, 16, 212, 212])
            visualize_attention_weights(attweight,visual_dir,filename,labels)
            break
        
# speech 2
# sound 2
# sing 2
# music 2

import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
from torch import Tensor
from matplotlib import font_manager
import numpy as np


def visualize_attention_weights(attweight, output_dir, filename, labels):
    num_layers = len(attweight)
    audio_index =  2  # batch 第几条可视化
    labels = labels[audio_index]
    if isinstance(labels, Tensor):
        labels = str(labels.item())
    output_dir = os.path.join(output_dir, filename[audio_index] + '_' + labels)   
    print(output_dir,'output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Adjusting the figure size for better clarity
    for layer_idx in range(num_layers):
        print(attweight[layer_idx].shape, 'attweight.shape')  # [batch, 16, 212, 212] 
        attention_matrix = attweight[layer_idx][audio_index].mean(dim=0).cpu().numpy()  
        print(attention_matrix.shape, 'attention_matrix.shape')  # [212, 212]
        
        # Increased figure size and font size
        plt.figure(figsize=(12, 6), dpi=300)
        
        # Plotting heatma
                 
        # # Copy the remaining columns from the original matrix
        # modified_matrix[:, 30:] = attention_matrix[:, 10:]


                    
        # Custom tick marks for 0, 50, 100, 150, 200, 211
        tick_y_positions = [0, 50, 100, 150, 211]
        tick_x_positions = [0, 50, 100, 150, 211]
        sns.heatmap(attention_matrix, cmap='viridis')
        print(attention_matrix.shape, 'attention_matrix.shape')  # [212, 212]
        
        tick_y_labels = [str(pos) for pos in tick_y_positions]
        tick_x_labels = [str(pos) for pos in tick_x_positions]
        plt.xticks(tick_x_positions, tick_x_labels, fontsize=24, rotation=0, weight='bold')
        plt.yticks(tick_y_positions, tick_y_labels, fontsize=24, weight='bold')
        
        # Increase colorbar font size and its tick labels
        colorbar = plt.gca().collections[0].colorbar
        colorbar.ax.tick_params(labelsize=20)
        font_properties = font_manager.FontProperties(weight='bold', size=20)
        for label in colorbar.ax.get_yticklabels():
            label.set_font_properties(font_properties)        
        # Adjust layout and save the image
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'layer_{layer_idx + 1}.png'), dpi=300)
        plt.close()



if __name__ == "__main__":
    args = init()
    # load model
    ckpt_path = os.path.join(args.model_path, "anti-spoofing_feat_model.pt")
    checkpoint = torch.load(ckpt_path)
    
    if args.model == "wpt-w2v2aasist":
        feat_model = WPTW2V2AASIST(model_dir= args.xlsr, prompt_dim=args.prompt_dim,
                                    num_prompt_tokens = args.num_prompt_tokens, num_wavelet_tokens=args.num_wavelet_tokens, 
                                    dropout= args.pt_dropout, visual=True).cuda()
    
    if args.model == 'pt-w2v2aasist':
        feat_model = PTW2V2AASIST(model_dir= args.xlsr, prompt_dim=args.prompt_dim, 
                                    num_prompt_tokens = args.num_prompt_tokens, dropout= args.pt_dropout, visual=True).cuda()    
    
    if args.model == 'ft-w2v2aasist':   
        feat_model = XLSRAASIST(model_dir= args.xlsr, freeze = False, visual=True).cuda()
        
    feat_model.load_state_dict(checkpoint)
    feat_model.eval()
    visual_dataset(feat_model,args)
    # visual_dataset_cam(feat_model,args)
    
    
    
    # # test on different dataset
    # if args.task == 'speech':
    #     test_on_speech(feat_model,args)
    # elif args.task == 'sound':
    #     test_on_sound(feat_model,args)
    # elif args.task == 'singing':
    #     test_on_singing(feat_model,args)
    # elif args.task == 'music':
    #     test_on_music(feat_model,args)
    # else:
    #     print('task not supported')
    #     exit(0)
        