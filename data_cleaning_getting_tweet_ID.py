#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the required libraries
import pandas as pd


# In[2]:


# read the tsv file 
# the orginal tsv file is over 9 GB; we didn't upload it to the github
tsv_read = pd.read_csv("full_dataset_clean.tsv", sep='\t')


# In[3]:


tsv_read.shape
# 209192029, 5
#total 209192029 tweets


# In[4]:


# english tweets total
tweets_english = tsv_read[tsv_read['lang'] == 'en']


# In[5]:


tweets_english.shape
#(118724580, 5)  total 118724580 tweets


# In[6]:


# random sample 1000 tweets out of the 118724580 english tweets

sample_tweets = tweets_english.sample(n=1000,replace=False,random_state=1)


# In[7]:


# sort the date
sample_tweets = sample_tweets.sort_values(by=['date'])


# In[8]:


# outout the whole dataframe to csv
sample_tweets.to_csv("sample_ID_info_1000.csv", index = False)
# output the tweet_id to csv for hydrator
sample_tweets['tweet_id'].to_csv("id_1000_sample.csv", index=False, header=False) 

# the two csvs are uploaded to the github
# next step is upload the id.csv file to Hydrator to get the hydrated tweets

