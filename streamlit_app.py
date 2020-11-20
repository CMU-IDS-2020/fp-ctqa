# Following architecture file copied from InferSent source code for load pretraining model
# In the later stage, this architecture will be modified to train a binary classifier (not 3 way softmax)
# We use the original code first to make sure the inference pipeline is good
# so that we don't have to worry about output capatibility when re-train this model with Quora

import numpy as np
import time
import pandas as pd
# import torch
# import torch.nn as nn
import streamlit as st
import urllib
import re
import pickle
import requests
import io
# import nltk
# nltk.download('punkt')
# import warnings
# warnings.filterwarnings("ignore")
st.title('Covid-19 Twitter Search')
def pd_load(inp):
    return pd.read_csv(inp)

def load_marix(inp):
    response = requests.get(inp)
    response.raise_for_status()
    all_scores = np.load(io.BytesIO(response.content))
    return all_scores

df = pd_load("https://raw.githubusercontent.com/CMU-IDS-2020/fp-ctqa/main/tweets.csv")
all_scores = load_marix('https://github.com/CMU-IDS-2020/fp-ctqa/raw/main/adjacency_matrix.npy')

tweets = df.text.tolist()
num_tweets = len(tweets)

st.subheader("Here are some randomly selected most recent tweets about Covid-19")
for i, tweet in enumerate(tweets[num_tweets - 10: num_tweets - 5]):
    st.write("[{}] ".format(i+1) + tweet)

def get_top_n_idx(A, N):
    N += 1 # not self
    col_idx = np.arange(A.shape[0])[:,None]
    sorted_row_idx = np.argsort(A, axis=1)[:,A.shape[1]-N::]
    best_scores = A[col_idx,sorted_row_idx]
    return sorted_row_idx, best_scores


sample_ids = [1,2,3,4,5]
#
# id_to_text = {}
# sample_ids = [1322748291646476288, 1322748291751227397, 1322748291839262723, 1322748292384645120, 1322748296432144385, 1322748297736454144, 1322748298797682690, 1322748301842747392, 1322748304946614273, 1322748305898573825]
# for n in sample_ids:
#     optimal_tweets = []
#     for i in range(5):
#         # optimal_tweets.append(get_t_value(n, index)) # This can be a function.
#         optimal_tweets.append(str(n) + " has its text as number " + str(i+1))
#     id_to_text[n] = optimal_tweets
# metric_choices = [id_to_text[name_id] for name_id in sample_ids]

st.subheader("Which tweet would you like to get information on?")
st.write("Please select the tweet id based on the number inside [] above")
tweet_option = st.selectbox('', sample_ids)
tweet_option -= 1
st.subheader("How many similar tweets would you like to retrieve?")
n_opt = st.slider("", min_value=1, max_value=5, value=3, step=1)


sorted_row_idx, best_scores = get_top_n_idx(all_scores[num_tweets - 10: num_tweets - 5], n_opt)
sorted_row_idx = sorted_row_idx[tweet_option].tolist()[::-1][1:]
best_scores = best_scores[tweet_option].tolist()[::-1][1:]


st.write('Here are the ordered top ' + str(n_opt) + ' tweets similar to this tweet:')

for tweet_idx, score in zip(sorted_row_idx, best_scores):
    st.write(tweets[tweet_idx])
    st.write("with similarity score " + str(score))
    st.write("\n")
