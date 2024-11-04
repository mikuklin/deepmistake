#!/bin/bash
ckpt_path=${1%/}
input_dir=${2%/}
theshold=${3%/}
output_dir=${4%/}

mkdir -p scores/${input_dir}/4class_pows/${output_dir}
cp -a ${input_dir}/. scores/${input_dir}/4class_pows/${output_dir}/

mkdir -p scores/${theshold}/4class_pows/${output_dir}
cp -a ${theshold}/. scores/${theshold}/4class_pows/${output_dir}/

python inference_deepmistake_part_2.py  --deepmistake_dir $ckpt_path --input_dir scores/${input_dir}/4class_pows/${output_dir} --treshold_dir scores/${theshold}/4class_pows/${output_dir} --mode 4class_pows
python evaluation_part_2.py scores/${input_dir}/4class_pows/${output_dir} scores/${input_dir}/4class_pows/${output_dir}

