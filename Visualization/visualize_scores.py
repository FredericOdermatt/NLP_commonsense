from visuals import Visualizor
import numpy as np
import pandas as pd
import os

# change name of desired model, e.g. JUSTERS
model_of_interest = 'KALM'
model_list = [model_of_interest+str(i+1) for i in range(3)]

execution_dir = os.path.dirname(os.path.abspath(__file__))

human_path =  execution_dir + "/../data100/human_eval_results.csv"
machine_path = execution_dir + "/../data100/machine_eval_results.csv"

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
col_names = machine_scoring.columns.values.tolist()
scores = [machine_scoring.iloc[:,i] for i in range(len(col_names))] + [human_avg_score]
names = col_names + ['Human mean']
vis.plot_joint(scores=scores, names=names)
#vis.plot_joint(scores=human_scores, names=human_names)