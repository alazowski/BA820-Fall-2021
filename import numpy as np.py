import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics 

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

import scikitplot as skplt

# color maps
from matplotlib import cm

## Load Dataset
df = pd.read_pickle("/Users/aleksanderlazowski/Desktop/GitHub/BA820-Fall-2021/assignments/assignment-01/forums.pkl")

## First Look at the Data
df.head(5)
df.info()

df.drop(columns="text", inplace=True)

#scaler = StandardScaler()
#df_scaled = scaler.fit_transform(df)
#df2 = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
#type(df2)

##First Cluster
cluster1 = linkage(df, method="complete")
cluster1

plt.figure(figsize=(15,5))
dendrogram(df)
plt.show()

## KMEANS
df.describe().T

k = KMeans(3, random_state=820)
k.fit(df)

k_labs = k.predict(df)
k_labs

k.n_iter_

df['k'] = k_labs
df.sample(3)

df.k.value_counts()