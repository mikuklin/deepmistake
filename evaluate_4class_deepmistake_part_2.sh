#!/bin/bash
ckpt_path=${1%/}
input_dir=${2%/}
output_dir=${3%/}

mkdir -p scores/${input_dir}/4class/${output_dir}
cp -a ${input_dir}/. scores/${input_dir}/4class/${output_dir}/

python inference_deepmistake_part_2.py  --deepmistake_dir $ckpt_path --input_dir scores/${input_dir}/4class/${output_dir} --mode 4class
python evaluation_part_2.py scores/${input_dir}/4class/${output_dir} scores/${input_dir}/4class/${output_dir}
