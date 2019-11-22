import numpy as np
from dtw import accelerated_dtw
import pandas as pd

# Store the data in a data frame
df = pd.read_csv('Data/Sales_Transactions_Dataset_Weekly.csv')

# Select the 52 weeks
data = df.iloc[:, 1:53].values

# Number of time series to compute
n = 40

# Extract n random time series
idx = np.random.choice(np.arange(len(data)), n, replace=False)

# Compute pairwise dtw distances
distances = []
for i in idx:
    for j in idx:
        distances.append(accelerated_dtw(data[i, :], data[j, :], 'euclidean')[0])
        
# Store the distances in an array
distances = np.array(distances)
distances = distances.reshape((n, n))

np.save('Data/distances', distances)
np.save('Data/idx', idx)