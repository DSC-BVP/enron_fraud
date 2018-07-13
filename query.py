#QUERY MODULE

#Imports
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class EmailData: 
  def __init__(self, data, vect):
    '''Instantiates class EmailData. Takes emails' body and vectorizer as arguments.'''
    self.vec = vect
    self.emails = data 
    #Train on the given email data.
    self.train()
  
  def train(self):
    '''Trains on emails' body using vectorizer.'''
    self.vec_train = self.vec.fit_transform(self.emails)
  
  def query(self, keyword, limit):
    '''Query using Cosine Similarity.'''
    vec_keyword = self.vec.transform([keyword])
    cosine_sim = cosine_similarity(vec_keyword, self.vec_train).flatten()
    related_email_indices = cosine_sim.argsort()[:-limit:-1]
    print(related_email_indices)
    return related_email_indices

  def find_email_by_index(self, i):
    return self.emails[i]
