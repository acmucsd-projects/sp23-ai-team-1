![](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/img/MBTI_Predictor.png)
(click the picture to access our website!)
# MBTI Classification from Tweets

[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-F3f0f0?&logo=Jupyter&labelColor=F3f0f0)](https://jupyter.org/try)
[![Python](https://img.shields.io/badge/Python-3.11.0-21455f?logo=python&labelColor=21455f)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.0-150458?logo=pandas&labelColor=150458)](https://pandas.pydata.org/pandas-docs/stable/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg?labelColor=blue)]([https://raw.githubusercontent.com/alckasoc/Joblisting-Webscraper/main/LICENSE](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/LICENSE))


## Table of Contents:
[1. Background](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#1-background)
[2. Dataset](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#2-dataset)
[3. Structure](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#3-structure)
[4. Requirements for Use](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#4-requirements-for-use)
[5. Technologies Used](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#5-technologies-used)
[6. Difficulties](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#5-difficulties)
[7. Author Info](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#6-author-info)

## 1. Background

MBTI, or Myers-Briggs Type Indicator, is a way to classy different personality types based on four categories: introversion(I)/extraversion(E), sensing(S)/intuition(N), thinking(T)/feeling(F), and judging(J)/perceiving(P). Every person has a combination of four letters that determines their personality types, and there are a total of 16 personality types. Although the MBTI tests are not known to be accurate and are mainly taken just for fun, we wanted to create a model that can predict a person’s MBTI based on their tweets.

## 2. Dataset

We decided to use the [MBTI Personality Type Twitter Dataset](https://www.kaggle.com/datasets/mazlumi/mbti-personality-type-twitter-dataset) from Kaggle as our data for our model. The dataset has almost 8000 values and contains tweets and their corresponding MBTI.

![image](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/img/Original_Dataset.png)
*Original Dataset* <br /> <br />

## 3. Structure

* `Meeting-Notes` store all of our past meeting notes
* `img` are the images included in our presentations
* `models` include our trained models
* `resources.md` is a list of all the resources we reference throughout this project

Note: the package versions listed in requirements.txt and imported in the code may not be the exact versions. However, the versioning here is less important. We've listed all of the used libraries in the section below.

## 4. Requirements for Use

* python
* pytorch
* nltk
* cleantext
* textblob
* transformers
* (to add)

## 5. Technologies Used

For the data cleaning and model training portion of our project, we used Google Colab, which is a collaborative application that allows multiple users to write Python code on the same file. We utilized NLTK, PyTorch, regex, and various libraries with the Colab to for our model. We also trained our model using BERT from PyTorch. To create our website, we used Streamlit, which is a Python-based library used to create web apps.


## 6. Difficulties

Since our project requires us to analyze tweets, the text in our dataset included tags, emojis, and other special characters. However, our model training process required us to remove these characters from the text. It was difficult figuring out how to take away the special characters without modifying the rest of the text.

We also realized that some of the tweets weren’t in English. This created problems with how we were tokenizing the text, so we had to decide between translating all of the non-English tweets into English, or deleting all of the non-English tweets. After we plotted the distribution of the language that all the tweets are in, we saw that the vast majority of the tweets are in English. As a result, we thought it would be best to remove all of the non-English tweets from our dataset.

<p align="center">
  <img src="https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/img/Language_Distribution.png">
  <i>Distribution of Languages</i>
</p>

Another hurdle we faced was figuring out how to train our model. We had trouble improving the accuracy of our model, and it took many hours before we achieved a low enough accuracy for our output. We were able to fix this problem by modifying our model and testing our various other models until we were satisfied with our results.

## 7. Author Info

- Vincent Tu (Advisor):            [LinkedIn](https://www.linkedin.com/in/vincent-tu-422b18208/) | [GitHub](https://github.com/alckasoc)
- Kevin Shen
- Yashil Vora
- Samuel Lee
- Vanessa Hu
- Chi Wong

