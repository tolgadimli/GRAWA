import pandas as pd
import numpy as np

NUM_SEEDS = 3

df = pd.read_csv('results8.csv')
m = int(len(df) / NUM_SEEDS)

dct = {}

for i in range(m):
    port = df[i*NUM_SEEDS:(i+1)*NUM_SEEDS]
    trace_mean, trace_std = np.mean(port['eigen_trace']), np.std(port['eigen_trace'])
    dct[port['dist_opt'].iloc[0]] = (trace_mean, trace_std)

print(dct)
    