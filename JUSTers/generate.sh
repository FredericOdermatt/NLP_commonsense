MODEL_TYPE=gpt2-medium

python run_generation.py \
--model_type $MODEL_TYPE \
--model_name_or_path $1 \
--length 128 \
--stop_token "<|endoftext|>" \
--k $2 \
--temperature $3 \
--p $4 \
--save_in_model_dir
