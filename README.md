![](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/img/MBTI_Predictor.png)
# MBTI Classification from Tweets

[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-F3f0f0?&logo=Jupyter&labelColor=F3f0f0)](https://jupyter.org/try)
[![Python](https://img.shields.io/badge/Python-3.11.0-21455f?logo=python&labelColor=21455f)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0.0-150458?logo=pandas&labelColor=150458)](https://pandas.pydata.org/pandas-docs/stable/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg?labelColor=blue)]([https://raw.githubusercontent.com/alckasoc/Joblisting-Webscraper/main/LICENSE](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/LICENSE))


## Table of Contents:
- [1. Background](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#1-background)
- [2. Getting Started](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#2-getting-started)
- [3. Structure](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#3-structure)
- [4. Requirements for Use](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#4-requirements-for-use)
- [5. Difficulties](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#5-difficulties)
- [6. Author Info](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/README.md#6-author-info)

## 1. Background

MBTI, or Myers-Briggs Type Indicator, is a way to classy different personality types based on four categories: introversion(I)/extraversion(E), sensing(S)/intuition(N), thinking(T)/feeling(F), and judging(J)/perceiving(P). Every person has a combination of four letters that determines their personality types, and there are a total of 16 personality types. Although the MBTI tests are not known to be accurate and are mainly taken just for fun, we wanted to create a model that can predict a person’s MBTI based on their tweets.

## 2. Getting Started

This is a project using NLP to classify a person's MBTI based on the tweets.

Our Dataset: [MBTI Personality Type Twitter Dataset](https://www.kaggle.com/datasets/mazlumi/mbti-personality-type-twitter-dataset)

If you're just getting started and want to learn the necessary tools going into this project, check out [resources.md](https://github.com/acmucsd-projects/sp23-ai-team-1/blob/main/resources.md)!

## 3. Structure

* `Meeting-Notes` store all of our past meeting notes
* `img` are the images included in our presentations
* `models` include our trained models
* `resources.md` is a list of all the resources we reference throughout this project

Note: the package versions listed in requirements.txt and imported in the code may not be the exact versions. However, the versioning here is less important. I've listed all used libraries.

## 4. Requirements for Use

* python
* pytorch
* nltk
* cleantext
* textblob
* transformers


## 5. Difficulties

We spent hours on cleaning and preprocessing the original data, as the tweets usually contain tags, emojis, and other special characters. In addition, we have tp try many different models such as BERT to improve our model performance. Finally, we worked on website, using Streamlit, and designed it to make it more attractive and informative.

## 6. Author Info

- Vincent Tu (Advisor):            [LinkedIn](https://www.linkedin.com/in/vincent-tu-422b18208/) | [GitHub](https://github.com/alckasoc)
- Kevin
- Yashil
- Samuel
- Vanessa
- Chi

