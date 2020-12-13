import numpy as np

from Scoring.Scores import MeteorScore, BLEUScore, RougeScore
from Scoring.Scores import BertScore, MoverScore
from Visualization.visuals import Visualizor
import pandas as pd

import os

execution_dir = os.path.dirname(os.path.abspath(__file__))
reference_path = execution_dir + "/Data/kalm_data/references/subtaskC_gold_answers.csv"
prediction_path = execution_dir + "/Data/kalm_data/predictions/subtaskC_answers.csv"

human_evaluated_only = True


## -------- Mover Score --------
MS = MoverScore(prediction_path, reference_path, human_eval=human_evaluated_only)
print("Mover Scores for similarity: ", np.mean(MS.scores))
MS.print_bad_results()

## -------- Rouge Score --------
RS = RougeScore(prediction_path, reference_path, human_eval=human_evaluated_only)
#RS = RougeScore(prediction_path, reference_path, rouge_type = ['rouge2',2]) 
# possible values for rouge_type[0] = ['rouge1',..., 'rouge9', 'rougeL']
# possible values for rouge_type[1] = [0,1,2], 0: recall (ROUGE), 1: precision (BLEU), 2: fmeasure
print("Rouge Scores for similarity: ", np.mean(RS.scores))
RS.print_bad_results()

## -------- BLEU Score --------
# chose score "own" or "challenge" uncomment print bad results
BS = BLEUScore(prediction_path, reference_path, which="own", human_eval=human_evaluated_only)
BS.print_bad_results()
print("BLEU Scores for similarity: ", np.mean(BS.scores))

## -------- BERT Score --------
# only works with GPU
BERTS = BertScore(prediction_path, reference_path, human_eval=human_evaluated_only)
print("BERT Scores for similarity: ", np.mean(BERTS.scores))
BERTS.print_bad_results()


## -------- METEOR Score --------
# does not work with GPU
METEORS = MeteorScore(prediction_path, reference_path, human_eval=human_evaluated_only)
print("METEOR Scores for similarity: ", np.mean(METEORS.scores))
METEORS.print_bad_results()

## -------- Human Score --------
human_path = execution_dir + "/Data/kalm_data/human_eval/KaLM-Eval.csv"
human_score = pd.read_csv(human_path)
test_ids = human_score["Input.sample_id"].unique()
human = np.empty((len(test_ids), 3))
for i in range(len(test_ids)):
    human[i] = human_score["Answer.reason1"].loc[human_score["Input.sample_id"] == test_ids[i]].to_numpy()
    noise1 = np.random.uniform(-0.15,0.15)
    noise2 = np.random.uniform(-0.15,0.15)
    noise3 = np.random.uniform(-0.15,0.15)
    human[i,0] += noise1
    human[i,1] += noise2
    human[i,2] += noise3

## Plotting results

vis = Visualizor()
vis.plot_hist(MS.scores, outfile_name="MS_hist")
vis.plot_hist(BS.scores, outfile_name="BS_hist")
scores = [MS.scores, BS.scores, RS.scores, human[:,0], human[:,1], human[:,2], np.mean(human, axis=1)]
names = ["Mover Score", "BLEU Score", "Rouge Score", "Human 1", "Human 2", "Human 3", "Human_mean"]
vis.plot_joint(scores=scores, names=names)

