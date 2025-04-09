# Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception

## ⭐I will finish the repository within a few weeks.

This is the repo of our work titled “Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for
Enhanced Auditory Perception”, which was available on arxiv at "".


We provided speech-trained WPT-XLSR-AASIST and final co-trained WPT-XLSR-AASIST pre-trained model, you can download from [google drive](https://drive.google.com/drive/folders/1h3w1anPb0k2GIuSfDG5JOvBNZmXLWhPn?usp=drive_link) and put them in `./ckpt_best`.


### 1. Data prepraring

This project requires downloading four datasets independently.

Speech - [ASVspoof2019](https://datashare.ed.ac.uk/handle/10283/3336)

Sound - [Codecfake-A3](https://zenodo.org/records/13838823)

Singing Voice - [CtrSVDD_train&dev](https://zenodo.org/records/10467648), [CtrSVDD_eval](https://zenodo.org/records/12703261)

Music - [FakeMusicCaps](https://zenodo.org/records/15063698)

Upon downloading all datasets, please arrange them in accordance with the directory structure outlined below. If any path errors occur, please modify the 'Data folder prepare' section in config.py accordingly.

```
# Project Directory Structure

## ASVspoof2019 Dataset
│   ├── ASVspoof2019
│   │   ├── LA
│   │   │   ├── ASVspoof2019_LA_train
│   │   │   │   └── flac
│   │   │   │        └── *.flac (25,380 audio files)
│   │   │   ├── ASVspoof2019_LA_dev
│   │   │   │   └── flac
│   │   │   │        └── *.flac (24,844 audio files)
│   │   │   ├── ASVspoof2019_LA_eval
│   │   │   │   └── flac
│   │   │   │        └── *.flac (71,237 audio files)
│   │   │   ├── ASVspoof2019_LA_cm_protocols
│   │   │   │   ├── ASVspoof2019.LA.cm.train.trn.txt (training labels)
│   │   │   │   ├── ASVspoof2019.LA.cm.dev.trl.txt (development labels)
│   │   │   │   ├── ASVspoof2019.LA.cm.eval.trl.txt (evaluation labels)

## CtrSVDD Dataset
│   ├── CtrSVDD
│   │   ├── train
│   │   │   └── *.wav (84,404 audio files)
│   │   ├── dev
│   │   │   └── *.wav (43,625 audio files)
│   │   ├── eval
│   │   │   └── *.wav (92,769 audio files)
│   │   ├── label
│   │   │   ├── train.txt (training labels)
│   │   │   ├── dev.txt (development labels)
│   │   │   ├── eval.txt (evaluation labels)

## Fakemusiccaps Dataset
│   ├── Fakemusiccaps
│   │   ├── audio
│   │   │   └── *.wav (33,041 audio files)
│   │   ├── label
│   │   │   ├── train.txt (training labels)
│   │   │   ├── dev.txt (development labels)
│   │   │   ├── eval.txt (evaluation labels)

## Fakesound Dataset
│   ├── Codecfake_A3
│   │   ├── 16kaudio
│   │   │   └── *.wav (99,112 audio files)
│   │   ├── label
│   │   │   ├── train.txt (training labels)
│   │   │   ├── dev.txt (development labels)
│   │   │   ├── eval.txt (evaluation labels)
```


### 2. Environment Setup
`conda create -n add python==3.9.18`

`pip install -r requirements.txt`


### 3. Training

Example: Speech-trained WPT-XLSRAASIST  
```
python main_train.py --gpu 0 --train_task speech --model wpt-w2v2aasist --batch_size 32 --o ./ckpt_pt/speech_wpt-w2v2aasist 
```

To change the training data, please refer to main_train.py `--train_task`, 

choices=["speech", "sound", "singing", "music", "cotrain"]

To change the CM, please refer to config.py `--model`, 

```
choices=['aasist', 'specresnet', 'fr-w2v2aasist','fr-wavlmaasist',  'fr-mertaasist',  ❄
          'ft-w2v2aasist','ft-wavlmaasist', 'ft-mertaasist',  🔥
          'pt-w2v2aasist', 'pt-wavlmaasist', 'pt-mertaasist', ⭐
          'wpt-w2v2aasist', 'wpt-wavlmaasist', 'wpt-mertaasist' ⭐⭐⭐
]
```
All training scripts for this paper can be found in `script/train_ref.sh`

### 4. Evaluation

You can use the checkpoint provided on our Google Drive for inference, and refer to the script `script/test_best.sh`

All inference scripts for this paper can be found in `script/test_ref.sh`

Compute EER score. This will iterate through all the result.txt files in the ckpt folder and return the EER scores.

`python evaluate_all.py -p ckpt_best/cotrain_wpt_xlsraasist `


### 5. Interpretability

You can generate the attention map using `script/visual.sh.`

Also, you can generate the T-SNE figure using `script/T-SNE.sh.`

## 📝 Citation

If you find this repository is useful to your research, please cite it as follows:


