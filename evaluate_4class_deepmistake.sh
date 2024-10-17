ckpt_path=$1
dir=$2

python inference_deepmistake.py  --deepmistake_dir $ckpt_path --input_dir $dir --mode 4class
mkdir -p scores/${dir}/4class/${ckpt_path}/
python evaluation.py $dir scores/${dir}/4class/${ckpt_path}/
