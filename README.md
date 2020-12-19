# CommonSense Reason Generation

In this repository we are looking at the task of generating reasons explaining why a statement is against common-sense.

I.e. given an input "He eats the submarine." the model should return something along the lines of "Submarines are not edible." or similar.

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
Install fairseq which is a submodule of the cloned gitrepo
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

# Training KaLM
```bash
bsub -o test.out -R "rusage[mem=8164,ngpus_excl_p=1]" -J first_test -W 4:00 <<< "NLP_commonsense/train_kalm.sh"
```

* -o: name of output file (should end in .out)
* -R: requirements for GPU
* -J: job name, useful for overview and to use bpeek
* -W: how much time is given to the job
* $1 model (JUSTers submission based on gpt2-medium)
* $2 batch_size (JUSTers: 64, however memory issue for cluster) 
* $3 per_gpu_train_batch_size (JUSTers: 5, however memory issue for cluster)
* $4 num_train_epochs (JUSTers: 5)

# Generate explanations based on own model

```bash
bsub -o test_gen.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J JUSTers_generate -W 4:00 ./generate.sh gpt2-medium 5 1 0.9
```
* $2 k (JUSTers: 50)
* $3 temperature (JUSTers: 1) 
* $3 p (JUSTers: 0.9)


# Evaluate predictions
Before running this script on the GPU, you should execute it on CPU first. This will download all needed pretrained models for the scoring methods. This might take several minutes. This has to be done only once and the GPU can be used afterwards. 

Inside the script, change the paths to your generated output and their reference files.

```bash
bsub -o score.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J evaluate_predictions -W 4:00 ./evaluate.sh 
```


# Evaluate Trained KaLM Model

Execute the following locally (not on the cluster). It allows you to interactively submit input sentences to the trained model and see the output.
```bash
./evaluate_kalm.sh $SCRATCH/KaLM/trained_models/checkpoint1.pt
...
2020-11-25 18:21:19 | INFO | fairseq_cli.interactive | Type the input sentence and press return:
The submarine is delicious.
...
Output: There is no way to be eaten in the sky.
```
# Training JUSTers

## Training on Google Colab

Open [JUSTers/Justers_colab.ipynb](https://colab.research.google.com/github/FredericOdermatt/NLP_commonsense/blob/master/JUSTers/Justers_colab.ipynb) directly in google colab by clicking on this link. After training on colab you can download the trained model using rsync as described in the notebook.

## Training on Leonhard

Note: Before submiting the job to the Leonhard cluster the training script must once be executed locally `./train.sh OUT_DIR_NAME 16 5 5`. This allows the script to download required models to the cache at ~/.cache where it can read it from when training on the GPU.
```bash
bsub -o test.out -R "rusage[mem=12000,ngpus_excl_p=1]" -J train_Justers -W 4:00 ./train.sh OUT_DIR_NAME 16 5 5
```

* -o: name of output file (should end in .out)
* -R: requirements for GPU
* -J: job name, useful for overview and to use bpeek
* -W: how much time is given to the job
* $1 output directory
* $2 batch_size (JUSTers: 64, however memory issue for cluster) 
* $3 per_gpu_train_batch_size (JUSTers: 5, however memory issue for cluster)
* $4 num_train_epochs (JUSTers: 5)

# Generate explanations based on own model

```bash
bsub -o test_gen.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J JUSTers_generate -W 4:00 ./generate.sh gpt2-medium 5 1 0.9
```
* $2 k (JUSTers: 50)
* $3 temperature (JUSTers: 1) 
* $3 p (JUSTers: 0.9)


# Evaluate predictions
Before running this script on the GPU, you should execute it on CPU first. This will download all needed pretrained models for the scoring methods. This might take several minutes. This has to be done only once and the GPU can be used afterwards. 

Inside the script, change the paths to your generated output and their reference files.

```bash
bsub -o score.out -R "rusage[ngpus_excl_p=1,mem=12000]" -J evaluate_predictions -W 4:00 ./evaluate.sh 
```


## Notes

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
