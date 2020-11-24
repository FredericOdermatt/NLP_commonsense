# Installation

This installation is for now made specifically for a node on Leonhard.
```bash
module load python_gpu/3.7.4
```

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

Install packages for scores
```bash
pip install moverscore
pip install nltk
```

To run the nltk tokenization open the python console and enter:
```bash
import nltk
nltk.download('punkt')
```

Execute 
```bash
bsub -o test.out -R "rusage[mem=8164,ngpus_excl_p=1]" -J first_test -W 4:00 <<< "NLP/KaLM/train3.sh"
```

* -o: name of output file (should end in .out)
* -R: requirements for GPU
* -J: job name, useful for overview and to use bpeek
* -W: how much time is given to the job

## Notes

* bjobs: lists current jobs
* bbjobs: lists current jobs, better overview
* bjobs -d: lists jobs that have finished a short while ago
* conda list: lists all installed packages in conda environment
* bpeek -J JOBNAME: will output recent lines a job wrote on the GPU
* An activated enviroment will automatically be picked up by the submission system.

## Working with submodules

The submodules are their own git-repo. Any change inside KaLM should be added and commit first inside KaLM. \
Then in a second step you can `git add KaLM` in the main folder and commit this change.
