#!/bin/bash
ckpt_path=$1
dir=$2
pip install krippendorff==0.8.0
python inference_deepmistake.py  --deepmistake_dir $ckpt_path --input_dir $dir
mkdir -p scores/${dir}/2class/${ckpt_path}/
python evaluation.py $dir scores/${dir}/2class/${ckpt_path}/
