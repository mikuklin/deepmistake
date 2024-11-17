grad_acc_steps=4
train_epochs=50
ft_epochs=50
DATA_FT_DIR=comedi_dataset_all_filtered_part_2_explode_filtered
ft_loss=crossentropy_loss_4_task_2
targ_emb=comb_dmn
hs=0
pool=mean
batch_norm=1
ft_save_by_score=spearman_disjudgement.average.score
linhead=$([ "$hs" == 0 ] && echo "true" || echo "false")

python run_model.py  --do_train --do_validation --data_dir $DATA_FT_DIR --output_dir WIC_DWUG+XLWSD+CoMeDi_all_ce_part_2_explode_filtered_comb_dmn --gradient_accumulation_steps $grad_acc_steps \
    --pool_type $pool --target_embeddings $targ_emb --head_batchnorm $batch_norm --loss $ft_loss --linear_head ${linhead} --head_hidden_size $hs \
    --num_train_epochs $ft_epochs --ckpt_path WIC_DWUG+XLWSD/ --log_train_metrics --save_by_score $ft_save_by_score