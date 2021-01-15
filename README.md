# CommonSense Reason Generation

In this repository we are looking at the task of generating reasons explaining why a statement is against common-sense.

I.e. given an input "He eats the submarine." the model should return something along the lines of "Submarines are not edible.".

We follow the challenge given in [SemEval 2020 Task C](https://competitions.codalab.org/competitions/21080#learn_the_details).


# Installation

```bash
git clone https://github.com/FredericOdermatt/NLP_commonsense
```

Inside NLP_commonsense install the submodules KaLM (our fork) and fairseq
```bash
git submodule update --init
```

To have clean environments we use conda, install miniconda from the official website
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

restart shell to make conda work after  `conda init`

Create an environment for the project
```bash
conda create -n nlp_env python=3.7.4
conda activate nlp_env
```
Install torch 1.4.0
```bash
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
```
Install fairseq which is a submodule of the cloned gitrepo.
```bash
pip install -e fairseq
```

Install other requirements
```bash
pip install -r requirements.txt
conda install --file requirements_conda.txt
```


To download the nltk extensions 'punkt' and 'wordnet' (for MeteorScore) execute the provided script.
```bash
chmod +x setup_nltk.sh
./setup_nltk.sh
```

# JUSTers

## Training on Google Colab - JUSTers

Google Colab has some GPUs that provide up to either 16 or even 25 Gb of GPU RAM. To train high batch-sizes we provide a python-notebook on google colab.

Open [JUSTers/Justers_colab.ipynb](https://colab.research.google.com/github/FredericOdermatt/NLP_commonsense/blob/master/JUSTers/Justers_colab.ipynb) directly in google colab by clicking on this link. After training on colab you can download the trained model using rsync as described in the notebook.

## Training on Leonhard - JUSTers

Before running this script on the GPU, you should execute it on CPU first. This will download all needed pretrained models for the scoring methods. This might take several minutes. This has to be done only once and the GPU can be used afterwards. 

The following training script only considers the data without evidence.
`./train.sh OUT_DIR_NAME 16 5 5`.
```bash
bsub -o test.out -R "rusage[mem=12000,ngpus_excl_p=1]" -J train_Justers -W 4:00 ./train.sh ${SCRATCH}/JUSTers/first_try 16 5 5
```

The following training script considers the data including evidence from Wiktionary. The current input format of the sentences during training is: "additional evidence <|evidence|> false-statement <|continue|> training target".
`./train.sh OUT_DIR_NAME 16 5 5`.
```bash
bsub -o test.out -R "rusage[mem=12000,ngpus_excl_p=1]" -J train_Justers -W 4:00 ./train_with_evidence.sh ${SCRATCH}/JUSTers/first_try 16 5 5
```

* $1 output directory
* $2 batch_size (JUSTers: 64, however memory issue for cluster) 
* $3 per_gpu_train_batch_size (JUSTers: 5, however memory issue for cluster)
* $4 num_train_epochs (JUSTers: 5)

To include additional evidence from Urban Dictionary change the commented section in the file finetune_envidence.py .

## Evaluation

To evaluate a desired model with the implemented scores use the executable evaluate.sh .
Within the executable change the arguments ref_path and pred_path to the corresponding reference and the prediction file containing the generated reasons of your model.
Further, set the bool for the desired metrics to be computed. Important to note is that MoverScore and BERTScore are only executable on GPU (as suggested in the command below). METEOR on the other hand is only executable on CPU. So its currently not possible to compute MoverScore together with METEOR in a single run. To combine all scores in a single .csv, first run the script with all metrics set to True besides METEOR. Then, run the script again, this time setting all scores to False besides METEOR.
```bash
bsub -o test.out -R "rusage[mem=12000,ngpus_excl_p=1]" -J ./evaluate.sh -W 4:00 ./evaluate.sh
```

## Visualization

First compute the automated scores of the generated outputs by executing running the above mentioned evaluate.sh script.
To create the scatter plot matrices along the correlation coefficients execute the file visualize_scores.py. This file uses the above created .csv and outputs a .png file with the matrix. GPU execution is not necessary.
```bash
python Visualization/visualize_scores.py
```

## Generate Explanations - JUSTers

```bash
bsub -o test_gen.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J JUSTers_generate -W 4:00 ./generate.sh PATH_TO_MODEL_FOLDER 5 1 0.9
```
* $1 path to folder containing model.bin etc.
* $2 k of TOP-K sampling (JUSTers: 50)
* $3 temperature (JUSTers: 1) 
* $3 p (JUSTers: 0.9)

# KaLM

## Training - KaLM
```bash
bsub -o test.out -R "rusage[mem=8164,ngpus_excl_p=1]" -J first_test -W 4:00 <<< "NLP_commonsense/train_kalm.sh"
```

* -o: name of output file (should end in .out)
* -R: requirements for GPU
* -J: job name, useful for overview and to use bpeek
* -W: how much time is given to the job

## Interactive Generation - KaLM

Execute the following locally (not on the cluster). It allows you to interactively submit input sentences to the trained model and see the output.
```bash
./evaluate_kalm.sh $SCRATCH/KaLM/trained_models/checkpoint1.pt
...
2020-11-25 18:21:19 | INFO | fairseq_cli.interactive | Type the input sentence and press return:
The submarine is delicious.
...
Output: There is no way to be eaten in the sky.
```

# Notes

* bjobs: lists current jobs
* bbjobs: lists current jobs, better overview
* bjobs -d: lists jobs that have finished a short while ago
* conda list: lists all installed packages in conda environment
* bpeek -J JOBNAME: will output recent lines a job wrote on the GPU
* An activated enviroment will automatically be picked up by the submission system.


## Working with submodules

The submodules are their own git-repo. Any change inside KaLM should be added, commited and pushed first inside KaLM. \
Then in a second step you can `git add KaLM` in the main folder and commit this change. To update submodules that were changed run
```bash
git submodule update
```
