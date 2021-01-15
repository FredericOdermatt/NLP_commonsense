python Scoring/compute_scores.py \
--ref_path  /./../data100/references_complete.csv \
--pred_path /./../data100/kalm.csv \
--call_BLEU False \
--call_ROUGE False \
--call_METEOR True \
--call_MoverScore False \
--call_BERTScore False \
--print_bad False

python Scoring/compute_scores.py \
--ref_path  /./../data100/references_complete.csv \
--pred_path /./../data100/ana.csv \
--call_BLEU False \
--call_ROUGE False \
--call_METEOR True \
--call_MoverScore False \
--call_BERTScore False \
--print_bad False

python Scoring/compute_scores.py \
--ref_path  /./../data100/references_complete.csv \
--pred_path /./../data100/butfit.csv \
--call_BLEU False \
--call_ROUGE False \
--call_METEOR True \
--call_MoverScore False \
--call_BERTScore False \
--print_bad False

python Scoring/compute_scores.py \
--ref_path  /./../data100/references_complete.csv \
--pred_path /./../data100/justers.csv \
--call_BLEU False \
--call_ROUGE False \
--call_METEOR True \
--call_MoverScore False \
--call_BERTScore False \
--print_bad False

bsub -R "rusage[ngpus_excl_p=1,mem=14000]" python Scoring/compute_scores.py \
--ref_path  /./../data100/references_complete.csv \
--pred_path /./../data100/kalm.csv \
--call_BLEU True \
--call_ROUGE True \
--call_METEOR False \
--call_MoverScore True \
--call_BERTScore True \
--print_bad True

bsub -R "rusage[ngpus_excl_p=1,mem=14000]" python Scoring/compute_scores.py \
--ref_path  /./../data100/references_complete.csv \
--pred_path /./../data100/ana.csv \
--call_BLEU True \
--call_ROUGE True \
--call_METEOR False \
--call_MoverScore True \
--call_BERTScore True \
--print_bad True

bsub -R "rusage[ngpus_excl_p=1,mem=14000]" python Scoring/compute_scores.py \
--ref_path  /./../data100/references_complete.csv \
--pred_path /./../data100/butfit.csv \
--call_BLEU True \
--call_ROUGE True \
--call_METEOR False \
--call_MoverScore True \
--call_BERTScore True \
--print_bad True

bsub -R "rusage[ngpus_excl_p=1,mem=14000]" python Scoring/compute_scores.py \
--ref_path  /./../data100/references_complete.csv \
--pred_path /./../data100/justers.csv \
--call_BLEU True \
--call_ROUGE True \
--call_METEOR False \
--call_MoverScore True \
--call_BERTScore True \
--print_bad True