import numpy as np

from Scoring.Scores import MoverScore, BLEUScore, RougeScore
from Visualization.visuals import Visualizor
import pandas as pd

import os

execution_dir = os.path.dirname(os.path.abspath(__file__))
reference_path = execution_dir + "/Data/kalm_data/references/subtaskC_gold_answers.csv"
prediction_path = execution_dir + "/Data/kalm_data/predictions/subtaskC_answers.csv"

human_evaluated_only = True

## -------- BLEU Score --------
# chose score "own" or "challenge" uncomment print bad results
BS = BLEUScore(prediction_path, reference_path, which="own", human_eval=human_evaluated_only)
BS.print_bad_results()
print(BS.scores)
print("BLEU Scores for similarity: ", np.mean(BS.scores))

