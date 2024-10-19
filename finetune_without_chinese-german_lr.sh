#!/bin/sh
LR=$1
grad_acc_steps=4
train_epochs=50
ft_epochs=50
DATA_FT_DIR=comedi_dataset_without_language/chinese-german
ft_loss=crossentropy_loss_4
targ_emb=dist_l1ndotn
hs=0
pool=mean
batch_norm=1
ft_save_by_score=krippendorff_alpha.average.score
linhead=$([ "$hs" == 0 ] && echo "true" || echo "false")

python run_model.py --learning_rate $LR --do_train --do_validation --data_dir $DATA_FT_DIR --output_dir WIC_DWUG+XLWSD+CoMeDi_without_chinese-german_4_lr$LR --gradient_accumulation_steps $grad_acc_steps \
    --pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $ft_loss --linear_head ${linhead} --head_hidden_size $hs \
    --num_train_epochs $ft_epochs --ckpt_path WIC_DWUG+XLWSD/ --log_train_metrics --save_by_score $ft_save_by_score
