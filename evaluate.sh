python test_scorer.py \
--ref_path  /JUSTers/data_dir/development-y.csv \
--pred_path /JUSTers/data_dir/development/3-gpt2-medium.csv \
--call_BLEU True \
--call_ROUGE True \
--call_METEOR False \
--call_MoverScore False \
--call_BERTScore True \
--print_bad False
