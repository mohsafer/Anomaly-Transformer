export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.6  --num_epochs 20   --batch_size 256  --mode train --dataset SMD  --data_path dataset/SMD   --input_c 38
python main.py --anormly_ratio 0.6 --num_epochs 10   --batch_size 256     --mode test    --dataset SMD   --data_path dataset/SMD     --input_c 38     --pretrained_model 20