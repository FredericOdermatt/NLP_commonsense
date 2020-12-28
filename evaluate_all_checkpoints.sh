# SCRIPT to evaluate all checkpoints

# Folder structure:
# -- BASE FOLDER
#  | final model
#  | -- checkpoint-1
#  |  | checkpoint-1 model
#  | -- checkpoint-2
# ....... and so on

NUMBER_OF_CHECKPOINTS=13

for ((i=1; i<=$NUMBER_OF_CHECKPOINTS; i++))
do
    bsub -o checkpoint${i}_eval_base.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J checkpoint${i}_eval -W 4:00 ./evaluate.sh /./../Data/kalm_data/references/subtaskC_gold_answers.csv /./../JUSTers/data_dir/pred_checkpoint-${i}.csv
done

# generation of base folder model
# bsub -o base_folder.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J base_folder -W 4:00 ./generate_with_evidence.sh ${BASE_FOLDER} 50 1 0.9
# echo "submitted generation for ${BASE_FOLDER}"
# /./../Data/kalm_data/references/subtaskC_gold_answers.csv
# /./../Data/justers/evaluation/evaluation-x.csv
## /./JUSTers/data_dir/pred_checkpoint-${i}.csv
