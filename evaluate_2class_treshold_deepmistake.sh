#!/bin/bash
ckpt_path=${1%/}
dir=${2%/}
theshold=${3%/}
output_dir=${4%/}

mkdir -p scores/${dir}/2class_treshold/${output_dir}
cp -a ${dir}/. scores/${dir}/2class_treshold/${output_dir}/

mkdir -p scores/${theshold}/2class_treshold/${output_dir}
cp -a ${theshold}/. scores/${theshold}/2class_treshold/${output_dir}/

python inference_deepmistake.py  --deepmistake_dir $ckpt_path --input_dir scores/${dir}/2class_treshold/${output_dir} --treshold_dir scores/${theshold}/2class_treshold/${output_dir} --mode 2class_treshold
python evaluation.py scores/${dir}/2class_treshold/${output_dir} scores/${dir}/2class_treshold/${output_dir}
