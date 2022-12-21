import pandas as pd
import numpy as np
import seaborn as sns

def visualize_heatmap(df):
    sns.heatmap(df.cpu(), cmap='Blues', annot=False)
