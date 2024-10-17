ckpt_path=$1
dir=$2
theshold=$3

python inference_deepmistake.py  --deepmistake_dir $ckpt_path --input_dir $dir --treshold_dir $theshold --mode 2class_treshold
mkdir -p scores/${dir}/2class_treshold/${ckpt_path}/
python evaluation.py $dir scores/${dir}/2class_treshold/${ckpt_path}/
