import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import torch
from model import BERTWithClassifierHead
from transformers import BertTokenizerFast
from torch import nn


st.set_page_config(page_title="Initial EDA")
st.sidebar.header("Initial EDA")

# Put graphs and EDA stuff here