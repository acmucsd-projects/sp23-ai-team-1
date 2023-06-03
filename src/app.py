import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import torch
from model import BERTWithClassifierHead
from transformers import BertTokenizerFast
from torch import nn


# RUN THE FILE- 
# python -m streamlit run app.py
# streamlit run app.py

st.title("ACM AI Projects- Spring '23- Team 1")

text = st.text_area("Input")

clicked = st.button('Generate MBTI Personality')

labels = ['intj', 'intp', 'entj', 'entp', 'infj', 'infp', 'enfj', 'enfp', 'istj', 'isfj', 'estj', 'esfj', 'istp', 'isfp', 'estp', 'esfp']   

@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BERTWithClassifierHead(num_classes=16)
    model.load_state_dict(torch.load('../models/mbti.pth', map_location=torch.device('cpu')))
    return model, tokenizer

model, tokenizer = load_model()
if clicked:
    res = tokenizer(text=text, 
                    padding='max_length', 
                    max_length=500, 
                    truncation=True, 
                    return_tensors='pt')
    m = nn.Softmax(dim=1)
    
    out = model(res)
    scaled = m(out)
    prediction = torch.argmax(scaled)
    st.write(labels[prediction])
    st.success("Data processed successfully! Here's your personality type: " + labels[prediction])


personality_type = ["intj", 'intp', 'entj', 'entp', 'infj', 'infp', 'enfj', 'enfp','istj', 'isfj', 'estj', 'esfj', 'istp', 'isfp', 'estp', 'esfp']
sample_prob = np.array([0.05, 0.05, 0.2, 0.1, 0.4, 0.99, 0.07, 0.4, 0.3, 0.2, 0.1, 0.65, 0.9, 0.7, 0.7, 0.1])

df = pd.DataFrame()
df["type"] = personality_type
df["prob"] = sample_prob

st.bar_chart(df, x="type", y="prob")

# st.bar_chart(sample_prob, x = personality_type)

num_epochs = ["1", "2", "3", "4"]
loss = np.array([5, 2.1, 1.5, 0.3])

# personality = "estj"
# st.error("Error in processing data. Please try again.")
