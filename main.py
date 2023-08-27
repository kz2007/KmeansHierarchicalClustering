import pandas as pd 
dataset=pd.read_csv("CC GENERAL.csv", sep=",")

dataset = dataset.drop(['CUST_ID'], axis=1)
for i in dataset.head() :
    dataset[i].fillna(dataset[i].mean(),inplace=True)

from sklearn.cluster import AgglomerativeClustering #Importing our clustering algorithm : Agglomerative
model=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
clusters=model.fit_predict(dataset)  #Applying agglomerative algorithm with 5 clusters, using euclidean distance as a metric

import matplotlib.pyplot as plt
scatter = plt.scatter(dataset['BALANCE'], dataset['PURCHASES'], c=clusters, cmap='rainbow')
plt.xlabel('BALANCE')
plt.ylabel('Clustered Data')
plt.colorbar(scatter)
plt.show()

from sklearn.cluster import KMeans  #Importing our clustering algorithm: KMeans
kmeans=KMeans(n_clusters=2, random_state=0)  #Cluster our data by choosing 5 as number of clusters
kmeans.fit(dataset)
kmeans.predict(dataset)
print(kmeans)
print(kmeans.cluster_centers_)
scatter = plt.scatter(dataset['BALANCE'], dataset['PURCHASES'], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label = 'Centroids')
plt.xlabel('BALANCE')
plt.ylabel('Clustered Data')
plt.colorbar(scatter)
plt.show()

sum_squared = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(dataset)
    sum_squared.append(km.inertia_)
plt.plot(K, sum_squared, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal K')
plt.show()
