#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the required libraries
import pandas as pd


# In[3]:


# read the tsv file 
# the orginal tsv file is over 9 GB; we didn't upload it to the github
tsv_read = pd.read_csv("full_dataset_clean.tsv", sep='\t')
# get the most recent 2000 tweets
tweets_data = tsv_read.tail(2000)
# filter the english tweets
tweets_english = tweets_data[tweets_data['lang'] == 'en']
# get the tweet_id and write it to csv file
tweets_english['tweet_id'].to_csv("id.csv", index=False, header=False) 
# the id.csv file is uploaded to the github
# next step is upload the id.csv file to Hydrator to get the hydrated tweets

