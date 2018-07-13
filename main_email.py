#MAIN MODULE


#Imports for data preparation and representation.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Imports for data manipulation and using models.
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
#Import user-defined functions.
from functions import *
from visual import Visual
from query import EmailData



#Data preparation:

#Read data and create a dataframe.
emails = pd.read_csv('emails.csv')
email_df = pd.DataFrame(parse_into_emails(emails.message))

# Drop emails with empty body, to or from_ columns. 
email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)

#Preview dataframe.
print("\nDataframe preview: \n",email_df.head())

#Print unique email addresses.
print("\nUnique FROM email addresses:",len(email_df.from_.unique()))
print("Unique TO email addresses:",len(email_df.to.unique()))



#Tokenize the bodies and convert them into a document-term matrix:

#Adding extra stop-words that appeared frequently in the dataset, but were not if interest.
stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])

#Vectorizer.
vect = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
X = vect.fit_transform(email_df.body)
features = vect.get_feature_names()

#Print the top terms across all documents.
print("\nMost frequent terms in the dataset: \n",top_mean_feats(X, features, None, 0.1, 10)) 



#Data classification:

#Applying Mini-KMeans because the dataset is large.
clf = MiniBatchKMeans(n_clusters=3, init_size=1000, batch_size=500, max_iter=100)  
clf.fit(X)
labels = clf.fit_predict(X)



#Data visualisation:

#Graphs by calling class Visual.
dv = Visual(X,labels,features,clf)
dv.raw_plot()
dv.cluster_plot()
dv.bargraph()



#Insights:

print("\nThe names PHILLIP and JOHN have come up numerous times, and thus it can be assumed that they had a strong relation with the workings of the company, and thus might have known about/orchestrated the fraud.\n")



#Querying on the insights:

print("Querying Phillip:\n")

#Instantiating class EmailData.
dq = EmailData(email_df.body, vect)

#Querying.
results = dq.query('phillip', 10)

#Print the first 10 results.
print(dq.find_email_by_index(results[:10]))
