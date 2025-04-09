# fr

python generate_score_all.py --gpu 3 --task speech --model_path ./ckpt/speech_fr-w2v2aasist &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/speech_fr-w2v2aasist &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/speech_fr-w2v2aasist &
python generate_score_all.py --gpu 6 --task music --model_path ./ckpt/speech_fr-w2v2aasist &

python generate_score_all.py --gpu 3 --task speech --model_path ./ckpt/speech_fr-mertaasist &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/speech_fr-mertaasist &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/speech_fr-mertaasist &
python generate_score_all.py --gpu 6 --task music --model_path ./ckpt/speech_fr-mertaasist &
wait

python generate_score_all.py --gpu 3 --task speech --model_path ./ckpt/speech_aasist &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/speech_aasist &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/speech_aasist &
python generate_score_all.py --gpu 6 --task music --model_path ./ckpt/speech_aasist &
wait

python generate_score_all.py --gpu 3 --task speech --model_path ./ckpt/speech_fr-wavlmaasist &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/speech_fr-wavlmaasist &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/speech_fr-wavlmaasist &
python generate_score_all.py --gpu 6 --task music --model_path ./ckpt/speech_fr-wavlmaasist &
wait

python generate_score_all.py --gpu 3 --task speech --model_path ./ckpt/speech_specresnet &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/speech_specresnet &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/speech_specresnet &
python generate_score_all.py --gpu 6 --task music --model_path ./ckpt/speech_specresnet &
wait


python generate_score_all.py --gpu 0 --task speech --model_path ./ckpt/singing_fr-w2v2aasist &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/singing_fr-w2v2aasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt/singing_fr-w2v2aasist & 
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/singing_fr-w2v2aasist &
wait 

python generate_score_all.py --gpu 0 --task speech --model_path ./ckpt/singing_fr-mertaasist &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/singing_fr-mertaasist &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/singing_fr-mertaasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/singing_fr-mertaasist &
wait

python generate_score_all.py --gpu 0 --task speech --model_path ./ckpt/singing_aasist &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/singing_aasist &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/singing_aasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/singing_aasist &
wait

python generate_score_all.py --gpu 0 --task speech --model_path ./ckpt/singing_fr-wavlmaasist &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/singing_fr-wavlmaasist &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/singing_fr-wavlmaasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/singing_fr-wavlmaasist &
wait

python generate_score_all.py --gpu 0 --task speech --model_path ./ckpt/singing_specresnet &
python generate_score_all.py --gpu 4 --task sound --model_path ./ckpt/singing_specresnet &
python generate_score_all.py --gpu 5 --task singing --model_path ./ckpt/singing_specresnet &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/singing_specresnet &
wait


python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt/sound_fr-w2v2aasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt/sound_fr-w2v2aasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt/sound_fr-w2v2aasist & 
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/sound_fr-w2v2aasist &
wait 


python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt/sound_fr-mertaasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt/sound_fr-mertaasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt/sound_fr-mertaasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/sound_fr-mertaasist &
wait

python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt/sound_aasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt/sound_aasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt/sound_aasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/sound_aasist &
wait

python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt/sound_fr-wavlmaasist &
python generate_score_all.py --gpu 3 --task sound --model_path ./ckpt/sound_fr-wavlmaasist &
python generate_score_all.py --gpu 3 --task singing --model_path ./ckpt/sound_fr-wavlmaasist &
python generate_score_all.py --gpu 3 --task music --model_path ./ckpt/sound_fr-wavlmaasist &
wait

python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt/sound_specresnet &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt/sound_specresnet &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt/sound_specresnet &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt/sound_specresnet &
wait

## ft
python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt_ft/singing_ft-w2v2aasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt_ft/singing_ft-w2v2aasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt_ft/singing_ft-w2v2aasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt_ft/singing_ft-w2v2aasist &
wait

python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt_ft/singing_ft-mertaasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt_ft/singing_ft-mertaasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt_ft/singing_ft-mertaasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt_ft/singing_ft-mertaasist &
wait

python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt_ft/singing_ft-wavlmaasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt_ft/singing_ft-wavlmaasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt_ft/singing_ft-wavlmaasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt_ft/singing_ft-wavlmaasist &
wait

python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt_ft/music_ft-w2v2aasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt_ft/music_ft-w2v2aasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt_ft/music_ft-w2v2aasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt_ft/music_ft-w2v2aasist &
wait

python generate_score_all.py --gpu 4 --task speech --model_path ./ckpt_ft/music_ft-mertaasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt_ft/music_ft-mertaasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt_ft/music_ft-mertaasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt_ft/music_ft-mertaasist &
wait

## pt
python generate_score_all.py --gpu 3 --task speech --model_path /ckpt_pt/speech_pt_10t-w2v2aasist 
python generate_score_all.py --gpu 4 --task sound --model_path /ckpt_pt/speech_pt_10t-w2v2aasist 
python generate_score_all.py --gpu 5 --task music --model_path /ckpt_pt/speech_pt_10t-w2v2aasist 
python generate_score_all.py --gpu 6 --task singing --model_path /ckpt_pt/speech_pt_10t-w2v2aasist 
