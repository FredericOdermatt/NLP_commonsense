# SCRIPT to evaluate all checkpoints

# Folder structure:
# -- BASE FOLDER
#  | final model
#  | -- checkpoint-1
#  |  | checkpoint-1 model
#  | -- checkpoint-2
# ....... and so on

BASE_FOLDER=${SCRATCH}/JUSTers/first_try
NUMBER_OF_CHECKPOINTS=4

for ((i=1; i<=$NUMBER_OF_CHECKPOINTS; i++))
do
    bsub -o checkpoint${i}.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J checkpoint${i} -W 4:00 ./generate.sh ${BASE_FOLDER}/checkpoint-${i} 50 1 0.9
    echo "submitted generation for ${BASE_FOLDER}/checkpoint-${i}"
done

# generation of base folder model
bsub -o base_folder.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J base_folder -W 4:00 ./generate.sh ${BASE_FOLDER} 50 1 0.9
echo "submitted generation for ${BASE_FOLDER}"