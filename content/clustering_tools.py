from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def clusterize(df_clean, data_clustered, min_df, max_df, n_clusters, min_label):
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

def remove_clusters(clusters, df_clean, data):
  data_clustered = data[~df_clean.label.isin(clusters)]
  df_clean['label'] = df_clean.label.apply(lambda x: clusters.index(x) if x in clusters else len(clusters))
  return data_clustered


def relevant_words_cluster(cluster, vectorizer, n=10):
    text = cluster.iloc[:, 1]
    vectors = vectorizer.transform(text).toarray()
    vectors = vectors.mean(axis=0)
    index = vectors.argsort()[-n:]
    relevant_words = []
    for word, key in vectorizer.vocabulary_.items():
        if key in index:
            relevant_words.append(word)
    return relevant_words