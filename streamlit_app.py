# Following architecture file copied from InferSent source code for load pretraining model
# In the later stage, this architecture will be modified to train a binary classifier (not 3 way softmax)
# We use the original code first to make sure the inference pipeline is good
# so that we don't have to worry about output capatibility when re-train this model with Quora

import numpy as np
import time
import torch
import torch.nn as nn
import streamlit as st
import urllib
import re
import pickle
import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")


id_to_text = {}
sample_ids = [1322748291646476288, 1322748291751227397, 1322748291839262723, 1322748292384645120, 1322748296432144385, 1322748297736454144, 1322748298797682690, 1322748301842747392, 1322748304946614273, 1322748305898573825]
for n in sample_ids:
    optimal_tweets = []
    for i in range(5):
        # optimal_tweets.append(get_t_value(n, index)) # This can be a function.
        optimal_tweets.append(str(n) + " has its text as number " + str(i+1))
    id_to_text[n] = optimal_tweets
metric_choices = [id_to_text[name_id] for name_id in sample_ids]

tweet_option = st.selectbox('Which tweet would you like to get information on?', sample_ids)
n_opt = st.selectbox('How many similar tweets would you like to retrieve?', (1, 2, 3, 4, 5))
st.write('Here are the top ' + str(n_opt) + ' tweets similar to this tweet:')
for i in range(n_opt):
    st.write(id_to_text[tweet_option][i])
# After all tasks
# del infersent.word_vec
