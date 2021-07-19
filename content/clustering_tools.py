from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
from typing import List, Tuple


def clusterize(df_clean, data_clustered, min_df, max_df, n_clusters, min_label) ->Tuple[TfidfVectorizer, KMeans, List[str]]:
  """
    cluster a sub-dataframe data_clustered of the dataframe df_clean using k-means. The content of the DataFrame is text converted using TfidfVectorizer
    New labels prediction are added to df_clean
    :param pd.DataFrame df_clean: dataframe with two columns: one for textes and the other for label
    :param pd.DataFrame data_clustered: contains text rows from df_clean
    :param float or int min_df: min_df parameter from TfidfVectorizer
    :param float or int max_df: min_df parameter from TfidfVectorizer
    :param int n_clusters: n_clusters parameter from KMean
    :param int min_label: labels from the clustering will be added to df_clean starting at value min_label+1
    :return Tuple[TfidfVectorizer, KMeans, List[str]]:  trained TfidfVectorizer to convert the text, trained KMean used for clustering, List of the words with highest mean tifidf score 
    within each cluster 
  """
  vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df, max_df=max_df)
  X = vectorizer.fit_transform(data_clustered)
  kmeans = KMeans(n_clusters=n_clusters, random_state=7).fit(X)
  if 'label' in df_clean.columns:
    df_clean.loc[df_clean.label>min_label, 'label'] = kmeans.labels_+min_label+1
  else:
    df_clean['label'] = kmeans.labels_
  relevant_words = []
  for i in range(len(df_clean.label.unique())):
      cluster = df_clean[df_clean.label==i]
      relevant_words.append(relevant_words_cluster(cluster, vectorizer, 20))
  return vectorizer, kmeans, relevant_words

def remove_clusters(clusters, df_clean, data) ->pd.DataFrame:
  """
    select data that are not labelled by index within the List clusters and convert the other label by len(clusters) in df_clean7
    :param List[int]: list of labels
    :param pd.DataFrame df_clean: DataFrame with a column label
    :param pd.DataFrame data: same DataFrame as df_clean without the column label
    :return pd.DataFrame: row from data that have label in clusters
  """
  data_clustered = data[~df_clean.label.isin(clusters)]
  df_clean['label'] = df_clean.label.apply(lambda x: clusters.index(x) if x in clusters else len(clusters))
  return data_clustered


def relevant_words_cluster(cluster, vectorizer, n=10)->List[str]:
    """
      return a list of the words with highest mean tifidf score within a dataframe
      :param pd.DataFrame cluster: DataFrame where 1 is a columns of strings
      :param TfidfVectorizer vectorizer: trained TfidfVectorizer to convert the string rows
      :param int n: lenght of the list to be returned
      :return List[str]: list of the words with highest mean tifidf score in cluster
    """
    text = cluster.iloc[:, 1]
    #turn text into tfidf-vectors and compute mean
    vectors = vectorizer.transform(text).toarray()
    vectors = vectors.mean(axis=0)
    #value of the indexes of highest tfidf-value
    index = vectors.argsort()[-n:]
    relevant_words = []
    for word, key in vectorizer.vocabulary_.items():
        if key in index:
            relevant_words.append(word) #the word that correspond to the index
    return relevant_words