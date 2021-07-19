# Document-Clustering

## Introduction
The Kleister Charity Dataset is dataset that contains PDF files published by British charities.
The dataset is available on [GitHub](https://github.com/applicaai/kleister-charity) and was extracted from https://www.gov.uk/government/organisations/charity-commission.

 As project during AI BeCode Training, We were asked to use clustering algorithms to give structure to this dataset and help a fictive company to sort the document.
 
## Sequenced Clustering

First, we reduce the number of words present in each text using Lemmatization techniques et removing english stop words. For this, we used spacy.

We transform texts into vectors using Tfidf. Then, we tried to cluster the dataset using KMeans. We limit ourself to this possibilities for two reasons: first, the set of hyperparameters
is simpler to tune. We basically tuned TfidfVectorizer parameters to get different vector dimensions and the cluster numbers to obtain different clustering. Second, it seems to us that topic
rather than meaning of the text was more important. For this it seems more reasonable to use Euclidean distance to compare points rather than for instance cosine-similarity.

The main issue using KMean was that most clusters were meaningless. Below, we can find a typical instance of clusters obtained using KMean and the most relevant words in each cluster (i.e. 
words with highest mean tfidf-score within each cluster):
 * Cluster 1 (size ~7%):
   'names', 'optional', 'information', 'choose', 'details', 'receipts', 'payments', 'accounts', 'procedures', 'funds', 'matters', 'including', 'objects'
 * Cluster 2 (size ~75%):
    'march', 'company', 'activities', 'statements', 'assets', 'accounts', 'limited', 'nas', 'funds', 'costs', 'matters', 'nat'
 * Cluster 3 (size ~10%):
   'august', 'school', 'scheme', 'statements', 'assets', 'accounts', 'funds', 'costs', 'governors', 'pupils', 'college', 'schools'
 * Cluster 4 (size ~8%):
   'december', 'church', 'activities', 'statements', 'assets', 'accounts', 'nas', 'funds', 'members', 'costs', 'pcc', 'matters', 'baptist', 'nat'
There are two issues with this clustering: first, cluster sizes are inequal (size of cluster 2 is way more larger than the other). Second, not all clusters have meaning : here, Cluster 3 and 4
have 'clear' thematic while Cluster 1 and 2 haven't.

To solve this issue, we decided to do sequential clustering: i.e. After a clustering like above, we keep the relevant Clusters (here 3 and 4) and regroup the other into a single set of remains.
Then, we cluster the set of remains further, keep relevant Clusters and so on.
At then end of the process, we end up with 14 clusters that are described in the next session. Note that the final number of Cluster and their content are somewhat arbitrary. Without feedback to 
determine the actual usefull clusters, it was difficult to make the choices in our process.



## Clusters Description
Here are each cluster I obtained and words with highest mean tfidf-score within:
 * School/College:
   'august', 'school', 'scheme', 'statements', 'assets', 'accounts', 'funds', 'costs', 'governors', 'pupils', 'college', 'schools'
 * Church
   'december', 'church', 'activities', 'statements', 'assets', 'accounts', 'nas', 'funds', 'members', 'costs', 'pcc', 'matters', 'baptist', 'nat'
 * Club
   'club', 'statements', 'assets', 'accounts', 'funds', 'members', 'costs', 'matters', 'rotary', 'ended', 'received', 'gymnastics', 'sailing', 'brought', 'records'
 * Art/Festival
    'theatre', 'arts', 'festival', 'statements', 'assets', 'accounts', 'limited', 'funds', 'costs', 'matters', 'including', 'ended', 'received', 'recognised', 'records'
 * Camp/scout
    'receipts', 'payments', 'county', 'leader', 'district', 'camp', 'leaders', 'procedures', 'expenses', 'matters', 'scout', 'scouts', 'ended', 'scouting', 'beavers', 'cubs'

 * University/research
    'university', 'research', 'continued', 'appointed', 'children', 'gains', 'members', 'students', 'matters', 'nhs', 'hospice', 'amounts', 'including', 'cancer', 'hospital'
 * International
    'projects', 'international', 'statements', 'assets', 'accounts', 'media', 'limited', 'africa', 'programmes', 'relief', 'funds', 'costs', 'women', 'including', 'received', 'ended', 'islamic'
 * Pension
    'housing', 'homes', 'services', 'care', 'scheme', 'pension', 'statements', 'assets', 'continued', 'defined', 'limited', 'funds', 'costs', 'including', 'received', 'properties', 'recognised', 'years'
 * River
   'policies', 'thames', 'river', 'investments', 'matters', 'directors', 'continued', 'prepared', 'amounts', 'leeds', 'received', 'losses', 'recognised', 'gains', 'ending'
 * Park/Museum
    'park', 'museum', 'policies', 'members', 'investments', 'expenses', 'matters', 'continued', 'prepared', 'expended', 'pages', 'valley', 'received', 'recognised', 'appointed', 'ended'
 * School/Optional
    'august', 'optional', 'school', 'funds', 'children', 'members', 'accounts', 'families', 'matters', 'playgroup', 'pre', 'events', 'needs', 'parents', 'received', 'pue', 'years'
 * Examination
    'examiner', 'examination', 'payments', 'community', 'funds', 'members', 'costs', 'accounts', 'assets', 'policies', 'expenses', 'resources', 'matters', 'statements', 'volunteers'
 * Liabililities/Policies
    'accounts', 'assets', 'liabilities', 'policies', 'members', 'funds', 'costs', 'december', 'resources', 'limited', 'continued', 'directors', 'prepared', 'statements', 'expended', 'amounts', 'losses'
 * Other (anything that don't fall into the previous clusters)
 
 
 
 

## Further improvements
 - Try to use cosine-similarity to classify document according to meaning and not only topics
 - Some clusters are much larger than the other or their content is less clear. One could use further clustering to get better clusters
 - This latter point is limited due to the context: in real situation, cluster should be defined according to the company need. As this project was mostly for exercise pur
