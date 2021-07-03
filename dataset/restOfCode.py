
import pandas as pd
import numpy as np
import scipy.linalg as la
# import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from sklearn.cluster import KMeans
from matplotlib import*
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
#from wpca import PCA
from sklearn.decomposition import PCA
import seaborn
from sklearn.svm import SVR


import warnings
warnings.filterwarnings("ignore")
# read data
df = pd.read_csv("/home/kawnayeen/PycharmProjects/Movie-Recommendation-System-Collaborative-filtering-MovieLens-100k-/abc.csv")


df.reset_index()
# df =df.drop(["Unnamed: 0"] ,  axis=1, inplace= True)
# print(df)

# convariance matrix
X_std = df.loc[:,'unknown':'western']
cov_mat = pd.DataFrame(X_std)
cov_mat = cov_mat.cov()
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# print('Eigenvalues in descending order:')
# for i in eig_pairs:
#     print(i[0])

# print(eig_vecs)

eigenVectors = pd.DataFrame(eig_vecs)
eigenVectors.to_csv("eigen.csv")

pca = PCA(n_components=3).fit(X_std)
XX = pca.transform(X_std)
print(XX)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# sort the eigenvalues in descending order
sorted_index = np.argsort(eig_vals)[::-1]
sorted_eigenvalue = eig_vals[sorted_index]

#genres
genres= ["unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
# sorting the coloumn names accordingly
genres_copy = genres.copy()
iter = 0;
for i in sorted_index:
    genres_copy[iter] = genres[i]
    iter = iter + 1


# similarly sort the eigenvectors
sorted_eigenvectors = eig_vecs[:, sorted_index]

# select the first n eigenvectors, n is desired dimension
# of our final reduced data.

n_components = 10  # you can select any number of components.
eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
X_reduced = np.dot(eigenvector_subset.transpose(),X_std.transpose()).transpose()

# Renaming coloumns
X_reduced = pd.DataFrame(XX)
# X_reduced.columns = genres_copy[0:n_components]
kmeanModel = KMeans(n_clusters=2)
kmeanModel.fit(X_reduced)

X_reduced_Cluster = X_reduced.copy()
X_reduced_Cluster['user_id'] = df["user_id"]
X_reduced_Cluster['Cluster'] = kmeanModel.labels_
pd.DataFrame(X_reduced_Cluster).to_csv("/home/kawnayeen/PycharmProjects/Movie-Recommendation-System-Collaborative-filtering-MovieLens-100k-/reduced_2.csv")

grouped = X_reduced_Cluster.groupby('Cluster')

i = 0

# for name,group in grouped:
#     X = X_reduced
#     Y = pd.DataFrame(columns=("rating"))
#     for i in (len(X_reduced)):
#         Y[i] = df.ratings


