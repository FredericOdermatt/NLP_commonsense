import numpy as np

from Scoring.Scores import MoverScore, BLEUScore, RougeScore
from Visualization.visuals import Visualizor

import os

execution_dir = os.path.dirname(os.path.abspath(__file__))
reference_path = execution_dir + "/Data/kalm_data/references/subtaskC_gold_answers.csv"
prediction_path = execution_dir + "/Data/kalm_data/predictions/subtaskC_answers.csv"


## -------- Mover Score --------
MS = MoverScore(prediction_path, reference_path)
print("Mover Scores for similarity: ", np.mean(MS.scores))
MS.print_bad_results()


## -------- BLEU Score --------
# chose score "own" or "challenge" uncomment print bad results
BS = BLEUScore(prediction_path, reference_path, which="own")
BS.print_bad_results()
print("BLEU Scores for similarity: ", np.mean(BS.scores))

## -------- Rouge Score --------
RS = RougeScore(prediction_path, reference_path)
#RS = RougeScore(prediction_path, reference_path, rouge_type = ['rouge2',2]) 
# possible values for rouge_type[0] = ['rouge1',..., 'rouge9', 'rougeL']
# possible values for rouge_type[1] = [0,1,2], 0: recall (ROUGE), 1: precision (BLEU), 2: fmeasure

print("Rouge Scores for similarity: ", np.mean(RS.scores))
RS.print_bad_results()


## Plotting results

vis = Visualizor()
vis.plot_hist(MS.scores, outfile_name="MS_hist")
vis.plot_hist(BS.scores, outfile_name="BS_hist")
scores = [MS.scores, BS.scores, RS.scores]
names = ["Mover Score", "BLEU Score", "Rouge Score"]
vis.plot_joint(scores=scores, names=names)
