#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the required libraries
import pandas as pd


# In[2]:


# read the tsv file 
# the orginal tsv file is over 9 GB; we didn't upload it to the github
tsv_read = pd.read_csv("full_dataset_clean.tsv", sep='\t')


# In[6]:


tsv_read.shape
#(205701547, 5)
#total 205701547 tweets


# In[7]:


# english tweets total
tweets_english = tsv_read[tsv_read['lang'] == 'en']


# In[8]:


tweets_english.shape
#(116714566, 5)  total 116714566 tweets


# In[11]:


# random sample 100,000 tweets out of the 116714566 english tweets

sample_tweets = tweets_english.sample(n=100000,replace=False,random_state=1)


# In[15]:


# sort the date
sample_tweets = sample_tweets.sort_values(by=['date'])


# In[17]:


# outout the whole dataframe to csv
sample_tweets.to_csv("sample_ID_info_100000.csv", index = False)
# output the tweet_id to csv for hydrator
sample_tweets['tweet_id'].to_csv("id_100000_sample.csv", index=False, header=False) 

# the two csvs are uploaded to the github
# next step is upload the id.csv file to Hydrator to get the hydrated tweets

