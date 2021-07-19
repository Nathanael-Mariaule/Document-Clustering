import pandas as pd
import spacy
import csv
from preprocessing import preprocessing
from clustering_tools import clusterize, remove_clusters
import pickle


#load the train set
train_set = '/kleister-charity/train/data'
df = pd.read_csv(train_set, quoting=csv.QUOTE_NONE, sep='\t', header=None)
data = df.iloc[:, 3].dropna()
#preprocess the third column using spacy
nlp = spacy.load("en_core_web_sm")
data = data.apply(lambda x: preprocessing(x, nlp))


#We will compute clusters using 9 KMean algorithms and selecting relevant labels each time
#parameter for clusterize to be used (e.g. parameters for TfidfVectorizer and KMean)
parameters = [
                [5, 0.8, 4,-1], 
                [5, 0.6, 8, 1], 
                [5, 0.4, 6, 3],
                [20, 0.3, 4, 4],
                [10, 0.6, 8, 5],
                [15, 0.3, 6, 7],
                [10, 0.6, 6, 9],
                [15, 0.7, 8, 10],
                [15, 0.6, 6, 11]
            ]
#labels selected for each KMean clustering
clusters_label = [
                    [2,3], 
                    [0,1,2,8],
                    [0,1,2, 3, 4],
                    [0,1,2, 3, 4, 7],
                    [0,1,2, 3, 4, 5, 8, 9],
                    [0,1,2, 3, 4, 5, 6, 7, 8, 9],
                    [0,1,2, 3, 4, 5, 6, 7, 8, 9, 10],
                    [0,1,2, 3, 4, 5, 6, 7, 8, 9, 10, 17],
                    [0,1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15]
                ]
#name of the final clusters
clusters_name = {
                    0 : 'School/College',
                    1 : 'Church',
                    2 : 'Club',
                    3 : 'Art/Festival',
                    4 : 'Camp/scout',
                    5 : 'University/research',
                    6 : 'International',
                    7 : 'Pension',
                    8 : 'River',
                    9 : 'Park/Museum',
                    10 : 'School/Optional',
                    11 : 'Examination',
                    12 : 'Liabililities/Policies',
                    13 : 'Other'
                }
cluster_relevant_words = []
vectorizers = []
kmeans = []

#clustering
df_clean = df[[0,3]].dropna()
vectorizer, kmean, relant_words = clusterize(df_clean, data.dropna(), *parameters[0])
vectorizers.append(vectorizer)
kmeans.append(kmean)
#for each cluster, we keep a list of the words with highest mean tfidf score within each cluster
for i in clusters_label[0]:
    cluster_relevant_words.append(relant_words[i])
for i in range(1, len(clusters_label)):
    clustered = remove_clusters(clusters_label[i-1], df_clean, data)
    vectorizer, kmean, relevant_words =  clusterize(df_clean, clustered, *parameters[i])
    vectorizers.append(vectorizer)
    kmeans.append(kmean)
    for j in clusters_label[i]:
        if j>= len(clusters_label[i-1]):
            cluster_relevant_words.append(relevant_words[j])

#save into a pickle
with open('model.pickel', 'wb') as file:
    pickle.dump([clusters_label, clusters_name, cluster_relevant_words, vectorizers, kmeans])
