#!/bin/bash
pip install krippendorff==0.8.0
ckpt_path=${1%/}
input_dir=${2%/}
output_dir=${3%/}

python inference_deepmistake.py  --deepmistake_dir $ckpt_path --input_dir $input_dir
mkdir -p scores/${input_dir}/2class/${output_dir}/
python evaluation.py $input_dir scores/${input_dir}/2class/${output_dir}/
