import streamlit as st
import pandas as pd
from PIL import Image

# Put graphs and EDA stuff here

st.set_page_config(page_icon="🔎")

df = pd.read_csv("../data/twitter_MBTI.csv")
description = df.describe()
df = df[["text", "label"]]

st.title("Exploratory Data Analysis")

st.header("Our Dataset")
st.markdown(
    """
    First 100 rows of our [dataset](https://www.kaggle.com/datasets/mazlumi/mbti-personality-type-twitter-dataset):
    """
)
st.dataframe(df.head(100), use_container_width=True)

st.header("Data Description")
st.markdown(
    """
    The dataset has 2 columns: text and label. For each row, text contains multiple tweets by a user and strings them together.
    The label column is the corresponding MBTI label for the text. There are 7811 columns and 2 rows in total with no missing data.
    """
)
st.table(description)

st.header("Label Distribution")
st.markdown(
    """
    The dataset contains more data favoring introverted personality types over extroverted personality types.
    INFP and INFJ are the most popular labels with 1282 and 1057 counts respectively. 
    ESTJ and ESTP are the least popular labels with 81 and 100 counts respectively.
    """
)
st.image(Image.open('./img/MBTI_Distribution.png'))

st.header("Word Count Distribution")
st.markdown(
    """
    The word counts for each row of text range from 300 to 4000 words.
    The distribution is skewed right with most texts being around 1000 words.
    """
)
st.image(Image.open('./img/WordCount_Distribution.png'))

st.header("Character Count Distribution")
st.markdown(
    """
    The character count distribution is pretty standard as the letter frequencies are proportional to their respective frequencies in the English language.
    The "other" category is high because many texts contain miscellaneous characters such as emojis and punctuation.
    """
)
st.image(Image.open('./img/CharCount_Distribution.png'))

st.header("Gibberish Probability Distribution")
st.markdown(
    """
    We decided to analyze the texts for gibberish in order to see if we should drop certain texts because they won't be useful for the training of our model.
    We found that all the texts were pretty normal with no texts containing complete gibberish.
    """
)
st.image(Image.open('./img/Gibberish_Distribution.png'))

st.header("Language Distribution")
st.markdown(
    """
    Since the some texts were in languages other than English, we decided to analyze the distributions of languages.
    We found that an overwhelming majority of the texts were in English, so we decided to drop the rows containing non-English texts.
    """
)
st.image(Image.open('./img/Language_Distribution.png'))