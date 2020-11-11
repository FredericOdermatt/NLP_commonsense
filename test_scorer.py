import pandas as pd
import numpy as np
import seaborn as sns
from Scoring.Scores import Scorer

scoring = Scorer()

answers_df = pd.read_csv("./KaLM/subtaskC_generated/subtaskC_answers.csv", index_col = 0)
references_df = pd.read_csv("~/NLP_project/SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALL_data/Test_Data/subtaskC_gold_answers.csv", index_col = 0)

if len(answers_df) != len(references_df):
    raise ValueError("Number of reference and generated reasons do not match.")

if any(answers_df.index.to_numpy() != references_df.index.to_numpy()):
    raise ValueError("Indices of provided answers and reference reasons do not match.")

answers_array = pd.concat([answers_df, answers_df, answers_df], axis=1).to_numpy()
references_array = references_df.to_numpy()
answers = answers_array.reshape((np.prod(answers_array.shape),)).tolist()
references = references_array.reshape((np.prod(answers_array.shape),)).tolist()

mover_scores = scoring.MoverScore(answers, references)
print("Mover Scores for similarity: ", np.mean(mover_scores))
moverscore_hist = sns.displot(mover_scores)
moverscore_hist.savefig("score_hist.png")

shown_elems = 3
idx = np.argpartition(mover_scores, shown_elems)
for i in idx[:shown_elems]:
    print("Reference:     ", references[i], "\n")
    print("Answer:     : ", answers[i], "\n")
    print("------------------------------------\n")
    