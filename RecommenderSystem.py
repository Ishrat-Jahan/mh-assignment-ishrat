import pandas as pd
import numpy as np
import scipy.linalg as la

# import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
from sklearn.cluster import KMeans
import scipy
from sklearn.svm import SVR
from pyeasyga import pyeasyga
from sklearn.metrics import silhouette_score
import random
import pyswarms as ps
from sklearn.neural_network import MLPRegressor


import warnings
warnings.filterwarnings("ignore")


from matplotlib import*
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
#from wpca import PCA
from sklearn.decomposition import PCA
import seaborn






global cluster_inertia_ga
cluster_inertia_ga = []
cluster_inertia_pso = []
cluster_inertia_random = []

mae_random = []
mae_ga = []
mae_pso= []

best_random = []
best_ga = []
best_pso = []




user_rating_matrix = np.zeros((944, 1683))
userIntX = pd.DataFrame
global n_components
# number of components to retain after PCA
n_components = 7
# number of clusters
n_clusters = 5
# discarded_list = []
#  A matrix that will hold the average rating for each user
global avg_rating


def get_PCA_user_matrix_X(fileName):
    # read data
    rating_df = pd.read_csv("./dataset/movieLens-100k/"+fileName, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

    item_df = pd.read_csv("./dataset/movieLens-100k/u.item", sep="|", encoding="latin-1",
                          names=["movie_id", "movie_title", "release_date", "video_release_date",
                                 "imdb_url", "unknown", "action", "adventure", "animation",
                                 "childrens", "comedy", "crime", "documentary", "drama", "fantasy",
                                 "film_noir", "horror", "musical", "mystery", "romance",
                                 "sci-fi", "thriller", "war", "western"])

    user_df = pd.read_csv("./dataset/movieLens-100k/u.user", sep="|", encoding="latin-1", names=["user_id", "age", "gender",
                                                                                        "occupation", "zip_code"])

    # convert timestamp column to time stamp
    rating_df["timestamp"] = rating_df.timestamp.apply(lambda x: datetime.fromtimestamp(x / 1e3))
    # drop unnecessary column
    item_df.drop(["video_release_date"],  axis=1, inplace= True)

    # merging user and rating matrix
    user_rating = pd.merge(user_df, rating_df)

    # mering user-rating and item matrix
    full_df = pd.merge(user_rating, item_df)

    # A matrix where rows stands for users and coloumns for movies and each item m_ij is the rating user i gave for movie j
    global user_rating_matrix
    user_rating_matrix = np.zeros((full_df.max(axis=0)['user_id']+1, full_df.max(axis=0)['movie_id']+1))

    # populate user-rating matrix from the full dataframe
    for i in range(len(full_df)):
        user_id = full_df.loc[i, 'user_id']
        movie_id = full_df.loc[i, 'movie_id']
        rating = full_df.loc[i, 'rating']
        user_rating_matrix[user_id][movie_id] = rating

    # saving this matrix for using later
    pd.DataFrame(user_rating_matrix).to_csv("user_rating.csv")


    # genres
    genres = ["unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama",
              "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]

    # grouping

    cols = ["user_id", "age", "gender", "occupation", "zip_code", "release_date", "avg_rating", "unknown", "action", "adventure",
            "animation",
            "childrens", "comedy", "crime", "documentary", "drama", "fantasy",
            "film_noir", "horror", "musical", "mystery", "romance",
            "sci-fi", "thriller", "war", "western"]

    df = pd.DataFrame(columns=cols)

    # grouping the full dataframe by each user that is extracting all rating information per user
    grouped = full_df.groupby('user_id')

    i = 0
    global avg_rating
    avg_rating = pd.DataFrame(columns=['user_id', 'avg_rating'])
    for name, group in grouped:

        # if (len(group) < 0):
        #     global discarded_list
        #     discarded_list.append(name)
        # else:
        avg_rating.at[i, "user_id"] = name
        avg_rating.at[i, "avg_rating"] = group['rating'].mean()

        df.at[i, "user_id"] = name
        df.at[i, "age"] = group["age"].to_list()[0]
        df.at[i, "gender"] = group["gender"].to_list()[0]
        df.at[i, "occupation"] = group["occupation"].to_list()[0]
        df.at[i, "zip_code"] = group["zip_code"].to_list()[0]
        dateDf = pd.to_datetime(group["release_date"]).values.astype(np.int64)
        df.at[i, "release_date"] = pd.to_datetime(dateDf.mean())
        df.at[i, "avg_rating"] = group['rating'].mean()

        for genre in genres:
            df.at[i, genre] = 0

            grouped2 = group[group[genre] == True].groupby(genre, as_index=False)
            for g, group2 in grouped2:
                df.at[i, genre] = group2["rating"].mean()
        i = i + 1

    # df.to_csv("df.csv")

    # Normalizing matrix - Only the genre columns
    min_max_scaler = preprocessing.MinMaxScaler()
    for genre in genres:
        df[genre] = min_max_scaler.fit_transform(df[[genre]])

    # Extracting only the genres columns
    X_std = df.loc[:, 'unknown':'western']

    # Applying PCA and plotting a scree graph to determine the elbow point
    pca = PCA().fit(X_std)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    # plt.show()

    pca = PCA(n_components=n_components).fit(X_std)
    x_reduced = pca.transform(X_std)
    reduced_matrix = pd.DataFrame(x_reduced)
    reduced_matrix['user_id'] = df["user_id"]
    reduced_matrix['avg_rating'] = df["avg_rating"]

    global userIntX
    userIntX = reduced_matrix.copy()
    userIntX.to_csv("userIntX_no_Cluster.csv")
    return reduced_matrix




def performClustering(algo):
    # global userIntX
    # XX = userIntX.copy()
    # XX.drop(['user_id', 'avg_rating'], axis=1, inplace=True)

    # reading from file
    XX = pd.read_csv("userIntX_no_Cluster.csv")
    global userIntX
    userIntX = XX.copy()
    XX.drop(['Unnamed: 0', 'user_id', 'avg_rating'], axis=1, inplace=True)

    min_max_scaler = preprocessing.MinMaxScaler()
    XX = pd.DataFrame(min_max_scaler.fit_transform(XX))

    if algo < 0:
        print("Kmeans++")
        kmeanModel = KMeans(n_clusters=n_clusters)
    else:
        if algo == 0:
            print("Random")
            centers = pd.DataFrame(XX).sample(n_clusters)
        elif algo == 1:
            print("GA")
            centers = run_genetic(XX)
        else:
            print("PSO")
            centers = pso()

        kmeanModel = KMeans(n_clusters=n_clusters, init=centers)

    kmeanModel.fit(XX)
    print(kmeanModel.inertia_ , silhouette_score(XX, kmeanModel.labels_))


    XX['user_id'] = userIntX['user_id']
    XX['Cluster'] = kmeanModel.labels_
    XX['avg_rating'] = userIntX['avg_rating']

    userIntX = XX.copy()
    XX.to_csv("userIntX.csv")

    # the following will portion if executed will give us the graph of cluster number vs inertia and
    # cluster number vs sl_score from where we can decide how many clusters to go with from the elbow points
    # of the graphs.


    # SSE = []
    # sl_score = []
    # for cluster in range(2, 20):
    #     kmeans = KMeans(n_clusters=cluster, init='k-means++')
    #     kmeans.fit(XX)
    #     print(cluster, kmeans.inertia_)
    #     SSE.append(kmeans.inertia_)
    #     sl_score.append(silhouette_score(XX, kmeans.labels_))
    #
    # # converting the results into a dataframe and plotting them
    # frame = pd.DataFrame({'Cluster': range(2, 20), 'SSE': SSE})
    # plt.figure(figsize=(12, 6))
    # plt.plot(frame['Cluster'], frame['SSE'], marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.show()
    #
    # frame = pd.DataFrame({'Cluster': range(2, 20), 'Sl_Score': sl_score})
    # plt.figure(figsize=(12, 6))
    # plt.plot(frame['Cluster'], frame['Sl_Score'], marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.show()


def get_prediction(testfilename, regression):
    rating_df = pd.read_csv("./dataset/movieLens-100k/"+testfilename, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

    prediction = []
    actual_rating = []

    if regression == True:
        print("REGRESSION: ")
    else:
        print("FORMULA: ")

    # print(len(discarded_list))
    for i in range(len(rating_df)):
        # if rating_df.loc[i, 'user_id'] not in discarded_list:
        actual_rating.append(rating_df.loc[i, 'rating'])
        if regression == True:
            prediction.append(predictRatingUsingRegression(rating_df.loc[i, 'user_id'], rating_df.loc[i, 'movie_id']))
        else:
            prediction.append(predictRatingUsingEquation(rating_df.loc[i, 'user_id'], rating_df.loc[i, 'movie_id']))

    calculate_error(np.array(actual_rating), np.array(prediction))


def calculate_error(y, yhat):
    d = y - yhat
    mse_f = np.mean(d ** 2)
    mae_f = np.mean(abs(d))
    rmse_f = np.sqrt(mse_f)

    print("Results by manual calculation:")
    print("MAE:", mae_f)
    print("RMSE:", rmse_f)


def predictRatingUsingEquation(user_id, movie_id):
    ux = pd.read_csv("userIntX.csv")
    ux.drop(['Unnamed: 0'], axis=1, inplace=True)

    ux = ux.set_index("user_id")

    # global avg_rating
    avgR = ux.at[user_id, 'avg_rating']

    clusterId = ux.at[user_id, 'Cluster']
    X = pd.DataFrame(ux.groupby('Cluster').get_group(clusterId))

    user_rating = pd.read_csv("user_rating.csv")
    user_rating.drop(['Unnamed: 0'], axis=1, inplace=True)
    user_rating_matrix = user_rating.values

    iter = 0
    sumDenom = 0
    sumNeum = 0
    for i, j in X.iterrows():
        corr = X.loc[user_id, :].corr(X.loc[i, :])

        if (user_rating_matrix[i][movie_id] != 0):
            r_i = user_rating_matrix[i][movie_id]
            r_avg = ux.at[i, 'avg_rating']
            sumDenom += corr
            sumNeum += corr*(r_i - r_avg)
            iter += 1

    if (sumDenom == 0):
        r_approx = avgR
    else:
        r_approx = avgR + (sumNeum/sumDenom)
    return r_approx
#
# def calc_rating():
#     ux = pd.read_csv("userIntX.csv")
#     ux.drop(['Unnamed: 0'], axis=1, inplace=True)
#
#     # global userIntX
#     # ux = pd.DataFrame(userIntX.copy())
#     ux = ux.set_index("user_id")
#
#
#
#     # clusterId = ux.at[user_id, 'Cluster']
#     # X = pd.DataFrame(ux.groupby('Cluster').get_group(clusterId))
#
#     user_rating = pd.read_csv("user_rating.csv")
#     user_rating.drop(['Unnamed: 0'], axis=1, inplace=True)
#     user_rating_matrix = user_rating.values
#
#
#
#     row, col = user_rating.shape
#
#     for  user in range(row):
#         for movie in range(col):
#             clusterId = ux.at[user, 'Cluster']
#             X = pd.DataFrame(ux.groupby('Cluster').get_group(clusterId))
#             if (user_rating_matrix[user][movie] == 0):
#                 sumDenom = 0
#                 sumNeum = 0
#                 for i, j in X.iterrows():
#                     corr = X.loc[user, :].corr(X.loc[i, :])
#                     if (user_rating_matrix[i][movie] != 0):
#                         r_i = user_rating_matrix[i][movie]
#                         r_avg = ux.at[i, 'avg_rating']
#                         sumDenom += corr
#                         sumNeum += corr * (r_i - r_avg)
#
#                 global avg_rating
#                 avgR = ux.at[user, 'avg_rating']
#
#                 if (sumDenom == 0):
#                     user_rating_matrix[user][movie] = avgR
#                 else:
#                     user_rating_matrix[user][movie] = avgR + (sumNeum / sumDenom)


def predictRatingUsingRegression(user_id, movie_id):

    ux = pd.read_csv("userIntX.csv")
    ux.drop(['Unnamed: 0'], axis=1, inplace=True)
    ux = ux.set_index("user_id")

    user_rating = pd.read_csv("user_rating.csv")
    user_rating.drop(['Unnamed: 0'], axis=1, inplace=True)
    user_rating_matrix = user_rating.values

    clusterId = ux.loc[user_id, 'Cluster']
    X = ux.groupby('Cluster').get_group(clusterId)
    y = pd.DataFrame(columns=['rating'])

    XX = X.copy()
    iter = 0
    for i, j in X.iterrows():
        if (user_rating_matrix[i][movie_id] != 0):
            y.at[iter, 'rating']= user_rating_matrix[i][movie_id]
            iter += 1
        else:
            XX.drop(i, inplace=True)

    XX.drop(['Cluster', 'avg_rating'], axis=1, inplace=True)

    user_vec = X.loc[user_id, :]
    user_vec.drop(['Cluster', 'avg_rating'], inplace=True)

    if (len(XX) == 0) :
        return ux.at[user_id, 'avg_rating']

    # regr = MLPRegressor(random_state=1, max_iter=500).fit(XX, y)
    # res = regr.predict([user_vec])
    # print(user_id, movie_id, res[0])
    # return res[0]
    regressor = SVR(C=1.0, epsilon=0.2)
    regressor.fit(XX, y)
    res2 = regressor.predict([user_vec])
    return res2[0]


####################==GA==####################

def create_individual(data):
    global n_clusters
    X= pd.DataFrame(data).sample(n_clusters)
    return X

def mutate(individual):
    noise = np.random.normal(0, .1, 1)
    l = len(individual)

    for i in range(l-1):
        index1 = random.randrange(0, 1)
        for j in range(l - 1):
            if index1 > 0.5 and 0 <= individual.iloc[i, j] + noise <= 1: # maintaining closure
                individual.iloc[i, j] += noise # adding some random noise



def fitness (individual, data):
    kmeanModel = KMeans(n_clusters=n_clusters, init=individual)

    kmeanModel.fit(data)
    global best
    score = silhouette_score(data, kmeanModel.labels_)
    # print(kmeanModel.inertia_, score)

    cluster_inertia_ga.append(kmeanModel.inertia_)
    return kmeanModel.inertia_


def crossover(parent_1, parent_2):
    iter = random.randrange(1, 3)


    # single point crossover
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    for i in range(iter):
        index1 = random.randrange(1, len(parent_1))
        index2 = random.randrange(1, len(parent_1))
        b, c = parent_1.iloc[index1].copy(), parent_2.iloc[index2].copy()
        child_1.iloc[index1], child_2.iloc[index2] = c, b


    # double point crossover
    l = len(parent_1)
    ind = random.randrange(1, l)


    for i in range(ind):
        index1 = random.randrange(0, l - 1)
        index2 = random.randrange(1, l - 1)
        child_1.iloc[index1, index2:l] = parent_2.iloc[index1, index2:l]
        child_2.iloc[index1, index2:l] = parent_1.iloc[index1, index2:l]


    return child_1, child_2


def run_genetic(data):
    # global userIntX
    # data = userIntX.copy()
    # data.drop(['user_id'], axis=1, inplace=True)
    # data = pd.read_csv("userIntX.csv")
    # data.drop(['Unnamed: 0', 'user_id'], axis=1, inplace=True)


    global best
    best = 0

    ga = pyeasyga.GeneticAlgorithm(data,
                                   population_size=10,
                                   generations=50,
                                   crossover_probability=0.7,
                                   mutation_probability=0.1,
                                   elitism=True,
                                   maximise_fitness=False)

    ga.create_individual = create_individual
    ga.mutate_function = mutate
    ga.fitness_function = fitness
    ga.crossover_function = crossover

    ga.run()
    (a, b) = ga.best_individual()
    print("GA Best:" , a)
    return b


####################==PSO==####################

def getIndv(arr):
    individual = np.zeros((n_clusters, n_components))
    j = 0;
    for i in range(len(individual)):
        individual[i, :] = arr[j:j+n_components]
        j += n_components
    return individual


def pso_fitness (arr):

    results = np.zeros(len(arr))

    for i in range(len(arr)):
        individual = pd.DataFrame(getIndv(arr[i, :]))
        kmeanModel = KMeans(n_clusters=n_clusters, init=individual)

        data = pd.read_csv("userIntX_no_Cluster.csv")
        data.drop(['Unnamed: 0', 'user_id', 'avg_rating'], axis=1, inplace=True)

        min_max_scaler = preprocessing.MinMaxScaler()
        data = pd.DataFrame(min_max_scaler.fit_transform(data))

        kmeanModel.fit(data)
        results[i] = kmeanModel.inertia_
    return results


def pso():
    max_bound = np.ones(n_clusters*n_components)
    min_bound = max_bound*0
    bounds = (min_bound, max_bound)
    options = {'c1': 0.4, 'c2': 0.5, 'w': 0.6}

    # Call instance of PSO with bounds argument
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=n_clusters*n_components, options=options, bounds=bounds)

    # Perform optimization
    cost, pos = optimizer.optimize(pso_fitness, iters=50)

    print("PSO ", cost, pos)
    return getIndv(pos)



# main

# data = pd.read_csv("userIntX_no_Cluster.csv")
# data.drop(['Unnamed: 0', 'user_id', 'avg_rating'], axis=1, inplace=True)
#
# min_max_scaler = preprocessing.MinMaxScaler()
# data = pd.DataFrame(min_max_scaler.fit_transform(data))
# run_genetic(data)
# get_model(2)
# get_prediction("my.test")




# get_PCA_user_matrix_X("u1.base")
# get_model(2)
# print(cluster_inertia_ga)
# x=[]
# for i in range(1, 501):
#     x.append(i)
#
# import matplotlib.pyplot as plot
#
# # plotting the points
# plot.plot(x, cluster_inertia_pso, label = "PSO-KM")
#
# plot.ylim(50,150)
# plot.xlim(1,100)
#
# plot.title('Comparison of Intertia')
#
# # show a legend on the plot
# plot.legend()
#
# # naming the x axis
# plot.xlabel('observation number')
# # naming the y axis
# plot.ylabel('Inertia')
#
# # giving a title to my graph
#
# # function to show the plot
# plot.show()

get_PCA_user_matrix_X("u1.base")

for i in range(-1,3):
    performClustering(i)
    get_prediction("my.test", False)







