python generate_score_all.py --gpu 3 --task speech --model_path ./ckpt_best/speech_wpt_xlsraasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt_best/speech_wpt_xlsraasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt_best/speech_wpt_xlsraasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt_best/speech_wpt_xlsraasist &
wait

python generate_score_all.py --gpu 3 --task speech --model_path ./ckpt_best/cotrain_wpt_xlsraasist &
python generate_score_all.py --gpu 5 --task sound --model_path ./ckpt_best/cotrain_wpt_xlsraasist &
python generate_score_all.py --gpu 6 --task singing --model_path ./ckpt_best/cotrain_wpt_xlsraasist &
python generate_score_all.py --gpu 7 --task music --model_path ./ckpt_best/cotrain_wpt_xlsraasist &
wait