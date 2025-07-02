python main_train.py\
    --image_dir D:\mwcl\code\R2GenCMN-main\data\iu_xray\images\
    --ann_path D:\ehr\code\R2GenCMN-main\data\iu_xray\annotation.json
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 100 \
    --batch_size 8 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 9233 \
    --beam_size 3 \
    --save_dir D:\mwcl\code\R2GenCMN-main\results\iu_xray\
    --log_period 50



python main_train.py --image_dir D:\mwcl\code\R2GenCMN-main\data\iu_xray\images\ --ann_path D:\mwcl\code\R2GenCMN-main\data\iu_xray\annotation.json --record_dir D:\ehr\code\R2GenCMN-main\records --dataset_name iu_xray --max_seq_length 60  --threshold 3 --epochs 100  --batch_size 6 --lr_ve 1e-4 --lr_ed 5e-4 --step_size 10 --gamma 0.8 --num_layers 3 --topk 32 --cmm_size 2048 --cmm_dim 512 --seed 7580 --beam_size 3 --save_dir D:\ehr\code\R2GenCMN-main\results\iu_xray\ --log_period 50 --n_gpu 1
--image_dir D:\mwcl\code\R2GenCMN-main\data\iu_xray\images\ --ann_path D:\mwcl\code\R2GenCMN-main\data\iu_xray\annotation.json --record_dir D:\ehr\code\R2GenCMN-main\records --dataset_name iu_xray --max_seq_length 60  --threshold 3 --epochs 100  --batch_size 6 --lr_ve 1e-4 --lr_ed 5e-4 --step_size 10 --gamma 0.8 --num_layers 3 --topk 32 --cmm_size 2048 --cmm_dim 512 --seed 7580 --beam_size 3 --save_dir D:\ehr\code\R2GenCMN-main\results\iu_xray\ --log_period 50 --n_gpu 1

