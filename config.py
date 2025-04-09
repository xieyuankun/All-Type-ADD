import argparse

def initParams():
    parser = argparse.ArgumentParser(description="Configuration for the project")

    parser.add_argument('--seed', type=int, help="Random number seed for reproducibility", default=688)

    # Train & Dev Data folder prepare 

    parser.add_argument("--asvspoof19_train_audio", type=str, help="Path to the training audio for ASVspoof2019 dataset",
                        default='yourpath/asv2019/LA/ASVspoof2019_LA_train/flac')
    parser.add_argument("--asvspoof19_train_label", type=str, help="Path to the training label for ASVspoof2019 dataset",
                        default="yourpath/asv2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")  
    parser.add_argument("--asvspoof19_dev_audio", type=str, help="Path to the development audio for ASVspoof2019 dataset",
                        default='yourpath/asv2019/LA/ASVspoof2019_LA_dev/flac')
    parser.add_argument("--asvspoof19_dev_label", type=str, help="Path to the development label for ASVspoof2019 dataset",
                        default="yourpath/asv2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")  
    parser.add_argument("--asvspoof19_eval_audio", type=str, help="Path to the evaluation audio for ASVspoof2019 dataset",
                        default='yourpath/asv2019/LA/ASVspoof2019_LA_eval/flac')   
    parser.add_argument("--asvspoof19_eval_label", type=str, help="Path to the evaluation label for ASVspoof2019 dataset",
                        default="yourpath/asv2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt") 

    parser.add_argument("--singfake_train_audio", type=str, help="Path to the training audio for CtrSVDD dataset",
                        default="yourpath/singfake/train/")
    parser.add_argument("--singfake_train_label", type=str, help="Path to the training label for CtrSVDD dataset",
                        default="yourpath/singfake/label/train.txt")
    parser.add_argument("--singfake_dev_audio", type=str, help="Path to the development audio for CtrSVDD dataset",
                        default="yourpath/singfake/dev/")
    parser.add_argument("--singfake_dev_label", type=str, help="Path to the development label for CtrSVDD dataset",
                        default="yourpath/singfake/label/dev.txt")
    parser.add_argument("--singfake_eval_audio", type=str, help="Path to the evaluation audio for CtrSVDD dataset",
                        default="yourpath/singfake/eval/")CtrSVDD
    parser.add_argument("--singfake_eval_label", type=str, help="Path to the evaluation label for CtrSVDD dataset",
                        default="yourpath/singfake/label/eval.txt")

    parser.add_argument("--fakemusiccaps_audio", type=str, help="Path to the audio for FakeMusicCaps dataset",
                        default="yourpath/Fakemusiccaps/audio/")
    parser.add_argument("--fakemusiccaps_train_label", type=str, help="Path to the training label for FakeMusicCaps dataset",
                        default="yourpath/Fakemusiccaps/label/train.txt")
    parser.add_argument("--fakemusiccaps_dev_label", type=str, help="Path to the development label for FakeMusicCaps dataset",
                        default="yourpath/Fakemusiccaps/label/dev.txt")    
    parser.add_argument("--fakemusiccaps_eval_label", type=str, help="Path to the evaluation label for FakeMusicCaps dataset",
                        default="yourpath/Fakemusiccaps/label/eval.txt") 

    parser.add_argument("--fakesound_audio", type=str, help="Path to the audio for Codecfake_A3 dataset",
                        default="yourpath/Codecfake_A3/16kaudio/")
    parser.add_argument("--fakesound_train_label", type=str, help="Path to the training label for Codecfake_A3 dataset",
                        default="yourpath/Codecfake_A3/label/train.txt")    
    parser.add_argument("--fakesound_dev_label", type=str, help="Path to the development label for Codecfake_A3 dataset",
                        default="yourpath/Codecfake_A3/label/dev.txt")      
    parser.add_argument("--fakesound_eval_label", type=str, help="Path to the evaluation label for Codecfake_A3 dataset",
                        default="yourpath/Codecfake_A3/label/eval.txt")  

    # SSL folder prepare
    parser.add_argument("--xlsr", default="yourpath/huggingface/wav2vec2-xls-r-300m/")
    parser.add_argument("--wavlm", default="yourpath/huggingface/wavlm-large/")
    parser.add_argument("--mert", default="yourpath/huggingface/MERT-300M/")
    
    
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')

    # countermeasure
    parser.add_argument("--audio_len", type=int, help="raw waveform length", default=64600)
    parser.add_argument('-m', '--model', help='Model arch', default='pt-w2v2aasist',
                        choices=['aasist','fr-w2v2aasist','fr-wavlmaasist', 'fr-mertaasist', 
                                 'specresnet','ft-w2v2aasist','ft-wavlmaasist', 'ft-mertaasist', 'pt-w2v2aasist', 'wpt-w2v2aasist', 
                                 'shpt-w2v2aasist','pt-wavlmaasist', 'wpt-wavlmaasist', 'pt-mertaasist', 'wpt-mertaasist'])
    
    # pt
    parser.add_argument("--prompt_dim", type=int, help="prompt dim", default=1024)
    parser.add_argument("--num_prompt_tokens", type=int, help="audio dim", default=10)
    parser.add_argument("--pt_dropout", type=float, help="dropout", default=0.1)
    
    # wpt
    parser.add_argument("--num_wavelet_tokens", type=int, help="wavelet token", default=4)
    
    return parser