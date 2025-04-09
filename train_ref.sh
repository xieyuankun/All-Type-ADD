##fr

# python main_train.py --gpu 0 --train_task speech --model aasist --lr 0.0001 --batch_size 24 --o ./ckpt/speech_aasist &
# python main_train.py --gpu 3 --train_task speech --model specresnet --o ./ckpt/speech_specresnet &
# python main_train.py --gpu 4 --train_task speech --model fr-w2v2aasist --o ./ckpt/speech_fr-w2v2aasist &
# python main_train.py --gpu 5 --train_task speech --model fr-wavlmaasist  --o ./ckpt/speech_fr-wavlmaasist &
# python main_train.py --gpu 6 --train_task speech --model fr-mertaasist  --o ./ckpt/speech_fr-mertaasist &

# python main_train.py --gpu 0 --train_task sound --model aasist --seed 1234 --lr 0.0001 --batch_size 24 --o ./ckpt/sound_aasist &
# python main_train.py --gpu 4 --train_task sound --model specresnet --o ./ckpt/sound_specresnet &
# python main_train.py --gpu 5 --train_task sound --model fr-w2v2aasist --o ./ckpt/sound_fr-w2v2aasist &
# python main_train.py --gpu 6 --train_task sound --model fr-wavlmaasist  --o ./ckpt/sound_fr-wavlmaasist &
# python main_train.py --gpu 7 --train_task sound --model fr-mertaasist  --o ./ckpt/sound_fr-mertaasist &

# python main_train.py --gpu 0 --train_task singing --model aasist --lr 0.0001 --batch_size 24 --o ./ckpt/singing_aasist &
# python main_train.py --gpu 4 --train_task singing --model specresnet --o ./ckpt/singing_specresnet &
# python main_train.py --gpu 5 --train_task singing --model fr-w2v2aasist --o ./ckpt/singing_fr-w2v2aasist &
# python main_train.py --gpu 6 --train_task singing --model fr-wavlmaasist  --o ./ckpt/singing_fr-wavlmaasist &
# python main_train.py --gpu 7 --train_task singing --model fr-mertaasist  --o ./ckpt/singing_fr-mertaasist &

# python main_train.py --gpu 3 --train_task music --model aasist --seed 1234 --lr 0.0001 --batch_size 24 --o ./ckpt/music_aasist &
# python main_train.py --gpu 4 --train_task music --model specresnet --o ./ckpt/music_specresnet &
# python main_train.py --gpu 5 --train_task music --model fr-w2v2aasist --o ./ckpt/music_fr-w2v2aasist &
# python main_train.py --gpu 6 --train_task music --model fr-wavlmaasist  --o ./ckpt/music_fr-wavlmaasist &
# python main_train.py --gpu 7 --train_task music --model fr-mertaasist  --o ./ckpt/music_fr-mertaasist &


##ft
# python main_train.py --gpu 7 --train_task speech --model ft-w2v2aasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/speech_ft-w2v2aasist
# python main_train.py --gpu 0 --train_task speech --model ft-mertaasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/speech_ft-mertaasist &
# python main_train.py --gpu 4 --train_task speech --model ft-wavlmaasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/speech_ft-wavlmaasist &
# python main_train.py --gpu 5 --train_task sound --model ft-w2v2aasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/sound_ft-w2v2aasist &
# python main_train.py --gpu 6 --train_task sound --model ft-mertaasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/sound_ft-mertaasist &
# python main_train.py --gpu 7 --train_task sound --model ft-wavlmaasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/sound_ft-wavlmaasist &

# python main_train.py --gpu 3 --train_task singing --model ft-w2v2aasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/singing_ft-w2v2aasist &
# python main_train.py --gpu 4 --train_task singing --model ft-mertaasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/singing_ft-mertaasist &
# python main_train.py --gpu 5 --train_task singing --model ft-wavlmaasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/singing_ft-wavlmaasist &
# python main_train.py --gpu 6 --train_task music --model ft-w2v2aasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/music_ft-w2v2aasist &
# python main_train.py --gpu 7 --train_task music --model ft-mertaasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/music_ft-mertaasist &
# python main_train.py --gpu 2 --train_task music --model ft-wavlmaasist --seed 1234 --batch_size 14 --lr 0.000001 --o ./ckpt_ft/music_ft-wavlmaasist &


##proposed pt
python main_train.py --gpu 2 --train_task speech --model pt-w2v2aasist --batch_size 32 --num_prompt_tokens 10 --o ./ckpt_pt/speech_pt_10t-w2v2aasist 
python main_train.py --gpu 3 --train_task sound --model pt-w2v2aasist --batch_size 32 --num_prompt_tokens 10 --o ./ckpt_pt/sound_pt_10t-w2v2aasist 
python main_train.py --gpu 4 --train_task music --model pt-w2v2aasist --batch_size 32 --num_prompt_tokens 10 --o ./ckpt_pt/music_pt_10t-w2v2aasist 
python main_train.py --gpu 5 --train_task singing --model pt-w2v2aasist --batch_size 32 --num_prompt_tokens 10 --o ./ckpt_pt/singing_pt_10t-w2v2aasist 
python main_train.py --gpu 6 --train_task cotrain --model pt-w2v2aasist --num_epochs 20 --interval 4 --batch_size 32 --num_prompt_tokens 10 --o ./ckpt_pt/cotrain_pt_10t-w2v2aasist

##proposed wpt
python main_train.py --gpu 7 --train_task speech --model wpt-w2v2aasist --batch_size 32 --num_prompt_tokens 6 --num_wavelet_tokens 4 --o ./ckpt_pt/speech_wpt-w2v2aasist 
python main_train.py --gpu 7 --train_task sound --model wpt-w2v2aasist --batch_size 32 --num_prompt_tokens 6 --num_wavelet_tokens 4 --o ./ckpt_pt/sound_wpt-w2v2aasist
python main_train.py --gpu 7 --train_task music --model wpt-w2v2aasist --batch_size 32 --num_prompt_tokens 6 --num_wavelet_tokens 4 --o ./ckpt_pt/music_wpt-w2v2aasist
python main_train.py --gpu 7 --train_task singing --model wpt-w2v2aasist --batch_size 32 --num_prompt_tokens 6 --num_wavelet_tokens 4 --o ./ckpt_pt/singing_wpt-w2v2aasist
python main_train.py --gpu 7 --train_task cotrain --model wpt-w2v2aasist --num_epochs 20 --interval 4 --batch_size 32 --num_prompt_tokens 6 --num_wavelet_tokens 4 --o ./ckpt_pt/cotrain_wpt-w2v2aasist 
