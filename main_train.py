import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from CSAM import *
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, Sampler
import torch.utils.data.sampler as torch_sampler
from backbone.rawaasist import *
from collections import defaultdict
from tqdm import tqdm, trange
from exp.feature_extraction_exp import *
from utils import *
import eval_metrics as em
from feature_extraction import *
import config
torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method('spawn', force=True)

def initParams():
    parser = config.initParams()
    # Training hyperparameters

    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=10, help="interval to decay lr")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="7")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers")
    
    parser.add_argument('--train_task', type=str, default="speech", choices=["speech", "sound", "singing", "music", "cotrain"],)
    parser.add_argument('--base_loss', type=str, default="ce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")

    # generalized strategy 
    parser.add_argument('--SAM', type= bool, default= False, help="use SAM")
    parser.add_argument('--ASAM', type= bool, default= False, help="use ASAM")
    parser.add_argument('--CSAM', type= bool, default= False, help="use CSAM")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds 
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))



        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            json.dump(vars(args), file, indent=4)  

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def shuffle(feat,  labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    labels = labels[shuffle_index]
    # this_len = this_len[shuffle_index]
    return feat, labels


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
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
        
        
        
        
        

    #feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  
    
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    
    if args.SAM or args.CSAM:
        feat_optimizer = torch.optim.Adam
        feat_optimizer = SAM(
            feat_model.parameters(),
            feat_optimizer,
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005
        )

    asvspoof19_trainset = asvspoof19(args.asvspoof19_train_audio,
                                        args.asvspoof19_train_label,
                                        audio_length=args.audio_len)
    
    asvspoof19_devset = asvspoof19(args.asvspoof19_dev_audio,
                                   args.asvspoof19_dev_label,
                                   audio_length=args.audio_len)

    codecfakesound_trainset = codecfakesound(args.fakesound_audio,
                                    args.fakesound_train_label,
                                    audio_length=args.audio_len)
    
    codecfakesound_devset = codecfakesound(args.fakesound_audio,
                                    args.fakesound_dev_label,
                                    audio_length=args.audio_len)
    singfake_trainset = singfake(args.singfake_train_audio,
                                    args.singfake_train_label,
                                    audio_length=args.audio_len)
    singfake_devset = singfake(args.singfake_dev_audio,
                                    args.singfake_dev_label,
                                    audio_length=args.audio_len)
    fakemusiccaps_trainset = fakemusiccaps(args.fakemusiccaps_audio,
                                    args.fakemusiccaps_train_label,
                                    audio_length=args.audio_len)    
    fakemusiccaps_devset = fakemusiccaps(args.fakemusiccaps_audio,
                                    args.fakemusiccaps_dev_label,
                                    audio_length=args.audio_len)
    
    if args.train_task == "speech":
        train_set = [asvspoof19_trainset]
        dev_set = [asvspoof19_devset]
    if args.train_task == "sound":
        train_set = [codecfakesound_trainset]
        dev_set = [codecfakesound_devset]
    if args.train_task == "singing":
        train_set = [singfake_trainset]
        dev_set = [singfake_devset]
    if args.train_task == "music":
        train_set = [fakemusiccaps_trainset]
        dev_set = [fakemusiccaps_devset]
    if args.train_task == "cotrain":
        train_set = [asvspoof19_trainset, codecfakesound_trainset, singfake_trainset, fakemusiccaps_trainset]
        dev_set = [asvspoof19_devset, codecfakesound_devset, singfake_devset, fakemusiccaps_devset]
    
    for dataset in train_set:
        print(len(dataset),f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."
    for dataset in dev_set:
        print(len(dataset),f"Dataset {dataset} length")
        assert len(dataset) > 0, f"Dataset {dataset} is empty. Please check the dataset loading process."

    training_set = ConcatDataset(train_set)
    
    validation_set = ConcatDataset(dev_set)


    trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size),
                            shuffle=False, num_workers=args.num_workers,
                            sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))))                     
    valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size),
                                    shuffle=False, num_workers=args.num_workers,
                                    sampler=torch_sampler.SubsetRandomSampler(range(len(validation_set))))


    trainOri_flow = iter(trainOriDataLoader)
    valOri_flow = iter(valOriDataLoader)

    if args.train_task == "speech":
        weight = torch.FloatTensor([10,1]).to(args.device)   
    if args.train_task == "sound":
        weight = torch.FloatTensor([1,1]).to(args.device)
    if args.train_task == "singing":
        weight = torch.FloatTensor([10,1]).to(args.device)
    if args.train_task == "music":                            #Class 0: 4298, Class 1: 16563
        weight = torch.FloatTensor([4,1]).to(args.device)
    if args.train_task == "cotrain":                         #Class 0: 53440, Class 1: 146583
        weight = torch.FloatTensor([3,1]).to(args.device)

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight)

    else:
        criterion = nn.functional.binary_cross_entropy()

    prev_loss = 1e8
    prev_eer = 1
    monitor_loss = 'base_loss'
  
    for epoch_num in tqdm(range(args.num_epochs)):

        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                feat, audio_fn,  labels = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                feat, audio_fn,  labels = next(trainOri_flow)
            labels = labels.to(args.device) 

            if args.SAM or args.ASAM or args.CSAM:
                enable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.mean().backward()
                feat_optimizer.first_step(zero_grad=True)

                disable_running_stats(feat_model)
                feats, feat_outputs = feat_model(feat)
                criterion(feat_outputs, labels).mean().backward()
                feat_optimizer.second_step(zero_grad=True)
            
            else:
                feat_optimizer.zero_grad()
                feats, feat_outputs = feat_model(feat)
                feat_loss = criterion(feat_outputs, labels)
                feat_loss.backward()
                feat_optimizer.step()


            trainlossDict['base_loss'].append(feat_loss.item())

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                            str(trainlossDict[monitor_loss][-1]) + "\n")

        feat_model.eval()
        with torch.no_grad():
            ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []
            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    feat, audio_fn, labels= next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    feat, audio_fn, labels= next(valOri_flow)
                labels = labels.to(args.device) 

                feats, feat_outputs = feat_model(feat)

                if args.base_loss == "bce":
                    feat_loss = criterion(feat_outputs, labels.unsqueeze(1).float())
                    score = feat_outputs[:, 0]
                else:
                    feat_loss = criterion(feat_outputs, labels)
                    score = F.softmax(feat_outputs, dim=1)[:, 0]

                ip1_loader.append(feats)
                idx_loader.append((labels))
                devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)

                desc_str = ''
                for key in sorted(devlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                # v.set_description(desc_str)
                print(desc_str)
            valLoss = np.nanmean(devlossDict[monitor_loss])
            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            print(scores[labels == 0].shape)
            print(scores[labels == 0])
            print(scores[labels == 1].shape)
            val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(np.nanmean(devlossDict[monitor_loss])) + "\t" + str(val_eer) +"\n")
            print("Val EER: {}".format(val_eer))

        if (epoch_num + 1) % 5 == 0:
            torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))

        if valLoss < prev_loss:

            torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            prev_loss = valLoss
        # if val_eer < prev_eer:
        #     # Save the model checkpoint
        #     torch.save(feat_model.state_dict(), os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
        #     prev_eer = val_eer



    return feat_model


if __name__ == "__main__":
    args = initParams()
    _, _ = train(args)
