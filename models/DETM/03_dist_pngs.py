import argparse
import torch
import pickle 
import numpy as np 
import os 
import math 
import random 
import sys
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.io

import pandas as pd


### data and file related arguments
parser = argparse.ArgumentParser(description='PLots....')

parser.add_argument('--num_topics', type=int, default=10, help='number of topics')
parser.add_argument('--cat', type=bool, default=True, help='number of topics')


args = parser.parse_args()

num_topics = args.num_topics
cat = args.cat


def plot_dist_before_2d(df, cat):
    
    sns.set_theme(style="whitegrid")
    if cat:
        g= sns.catplot(x='topic', y='year',data=df)
    else:
        g = sns.boxenplot(y='year', x='topic', data=df)
    g.set(xlabel= 'Topic', ylabel='Year')

    plt.title(f'The Distribution of Tweets with {num_topics} Topics Before Filtering')
    plt.savefig(f'/home/yiyi/nlp_tm/models/DETM/k_{num_topics}/dist_before_filtering_time_{num_topics}_cat_{cat}.png', bbox_inches='tight')
    
def plot_dist_after_2d(df, cat):
    df['relevant'] = df['cossim']>0
    sns.set_theme(style="whitegrid")
    if cat:
        g= sns.catplot(x='topic', y='year',data=df[df['relevant']==True])
    else:
        g = sns.boxenplot(y='year', x='topic', data=df[df['relevant']==True])
    g.set(xlabel= 'Topic', ylabel='Year')
    plt.title(f'The Distribution of Tweets with {num_topics} Topics After Filtering')
    plt.savefig(f'/home/yiyi/nlp_tm/models/DETM/k_{num_topics}/dist_after_filtering_time_{num_topics}_cat_{cat}.png', bbox_inches='tight')
    

def count_dist(df, log=True):
    plt.figure(figsize=(10,25))
    sns.set_theme(style="white")
    df['relevant'] = df['cossim']>0
    g= sns.catplot(x="topic", hue="relevant", data=df, kind="count")
    plt.legend(loc='upper right', title='Relevant')
    g.set(xlabel= 'Topic', ylabel='Count')
    if log:
        plt.yscale('log')
    plt.title(f'The Distribution of Relevant or Not Relevant Tweets with {num_topics} Topics')
    plt.savefig(f'/home/yiyi/nlp_tm/models/DETM/k_{num_topics}/dist_bef_after_{num_topics}_log_{log}.png', bbox_inches='tight')
    
    
def plot_cos(df):
    plt.figure(figsize=(12,10))
    df['text'].groupby(pd.cut(df['cossim'], 20)).count().plot(kind='bar')
    plt.xlabel('Confidence Score (Cosine)')
    plt.ylabel('Count')
    plt.title(f'Distribution of Confidence Score with {num_topics} Topics')
    plt.savefig(f'/home/yiyi/nlp_tm/models/DETM/k_{num_topics}/dist_cossim_{num_topics}.png', bbox_inches='tight')
    
    
def main():
    
    file  = f'/home/yiyi/nlp_tm/models/DETM/k_{num_topics}/df_detm_topics_k{num_topics}_confidence.csv'
    print('read file ', file)
    
    df = pd.read_csv(file, index_col=0)
    
    
    plot_dist_before_2d(df, cat)
    plot_dist_after_2d(df, cat)
    count_dist(df, log=True)
    count_dist(df, log=False)
#         plot_cos(df)

    
    
    
if __name__ == '__main__':
    main()