python Scoring/compute_scores.py \
--ref_path  $1 \
--pred_path $2 \
--call_BLEU True \
--call_ROUGE True \
--call_METEOR False \
--call_MoverScore True \
--call_BERTScore True \
--print_bad False
