from visuals import Visualizor
import numpy as np
import pandas as pd
model_of_interest = 'KALM'
model_list = [model_of_interest+str(i+1) for i in range(3)]

human_path =  "./../data100/human_eval_results.csv"
machine_path = "./../data100/machine_eval_results.csv"

human_scoring = pd.read_csv(human_path, header=0)
machine_scoring = pd.read_csv(machine_path, header=0)
human_avg_score = human_scoring[model_list].mean(axis=1)
human_ind_scores = human_scoring[model_list]

'''
# computation of noise should be added after computing the correlation
human_names = ['Human 1', 'Human 2', 'Human 3']
human_scores = [human_ind_scores.iloc[:,0].astype('float'), human_ind_scores.iloc[:,1].astype('float'), human_ind_scores.iloc[:,2].astype('float')]
noise_level = 0.15
for i in range(len(human_scores[0])):
    noise1 = np.random.uniform(-1*noise_level,noise_level)
    noise2 = np.random.uniform(-1*noise_level,noise_level)
    noise3 = np.random.uniform(-1*noise_level,noise_level)
    human_scores[0][i] += noise1
    human_scores[1][i] += noise2
    human_scores[2][i] += noise3
'''
vis = Visualizor()
#vis.plot_hist(MS.scores, outfile_name="MS_hist")
#vis.plot_hist(BS.scores, outfile_name="BS_hist")
scores = [machine_scoring['BLEU'], machine_scoring['ROUGE'], machine_scoring['MOVER'], machine_scoring['BERT'], machine_scoring['METEOR'], human_avg_score]
names = machine_scoring.columns.values.tolist() + ['Human mean']
#vis.plot_joint(scores=scores, names=names)
vis.plot_box(scores=scores, names=names)
#vis.plot_joint(scores=human_scores, names=human_names)