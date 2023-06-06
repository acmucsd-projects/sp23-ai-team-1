import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import torch
from model import BERTWithClassifierHead
from transformers import BertTokenizerFast
from torch import nn
from PIL import Image
import os


# RUN THE FILE- 
# python -m streamlit run app.py
# streamlit run app.py

st.set_page_config(
    page_title="Welcome to our app!"
)

st.sidebar.header("Our Model")

st.title("ACM AI Projects- Spring '23- Team 1")

st.sidebar.success("Want to learn more about our project?")

path = os.path.dirname(__file__)
my_file1 = path+'/mbti1.jpg'


image1 = Image.open(my_file1)

st.markdown(
    """
    The MBTI Personality Test is a popular test used to determine someone's 
    personality based on their answers to a number of questions. We have used a 
    dataset which determines one's personality based on their tweets and built a 
    neural network to train our model with this data and allow any user to input 
    their tweets for our model to give an output of their MBTI personality. 
    
    Feel free to try different input tweets and play around with the model! 
    """
)

st.image(image1)

st.markdown(
    """
    ### Test Our Model
    """
)
text = st.text_area("Input tweet")

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


    df = pd.DataFrame()
    scaled = scaled[0].detach().numpy()
    df["type"] = labels
    df["prob"] = scaled

    
    path = os.path.dirname(__file__)
    my_file2 = path+'/mbti2.jpeg'
    image2 = Image.open(my_file2)

    st.image(image2, caption='What your personality means')

    st.bar_chart(df, x="type", y="prob")
    st.balloons()
