#!/bin/bash
ckpt_path=${1%/}
dir=${2%/}
output_dir=${3%/}
python inference_deepmistake.py  --deepmistake_dir $ckpt_path --input_dir $dir --mode 4class
mkdir -p scores/${dir}/4class/${output_dir}/
python evaluation.py $dir scores/${dir}/4class/${output_dir}/
