python run_generation.py \
--model_type gpt2-medium \
--model_name_or_path models_dir/$1 \
--length 128 \
--stop_token "<|endoftext|>" \
--k $2 \
--temperature $3 \
--p $4 
