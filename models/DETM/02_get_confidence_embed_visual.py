from __future__ import print_function

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

from itertools import chain

import data 

from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from detm import DETM
from utils import nearest_neighbors, get_topic_coherence
from sklearn.manifold import TSNE

import pandas as pd
import plac

from sklearn import cluster
from sklearn import metrics
from scipy.spatial import distance

from numpy import dot
from numpy.linalg import norm



parser = argparse.ArgumentParser(description='Get confidence from centroid....')

### data and file related arguments
parser.add_argument('--model_name', type=str, default='results/detm_twitter_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_val_ppl_2638.9_epoch_20', help='Path of the pre-trained model...')
parser.add_argument('--num_times', type=int, default=8, help='number of years')
parser.add_argument('--num_topics', type=int, default=50, help='number of years')
parser.add_argument('--get_confidence', type=bool, default=True, help='whether to get confidence of tweets')
args = parser.parse_args()


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

root_dir = '/home/yiyi/nlp_tm/'

def cosine_similarity(list_1, list_2):
    cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
    return cos_sim

def load_data(input_dir= os.path.join(root_dir, 'preprocessed_data')):
    token_file = os.path.join(input_dir, 'bow_tokens')
    count_file = os.path.join(input_dir, 'bow_counts')
    time_file = os.path.join(input_dir, 'bow_timestamps')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    times = scipy.io.loadmat(time_file)['timestamps'].squeeze()
    
    return tokens, counts, times


def _eta_helper(model, rnn_inp):
    inp = model.q_eta_map(rnn_inp).unsqueeze(1)
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    output = output.squeeze()
    etas = torch.zeros(model.num_times, model.num_topics).to(device)
    inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
    etas[0] = model.mu_q_eta(inp_0)
    for t in range(1, model.num_times):
        inp_t = torch.cat([output[t], etas[t-1]], dim=0)
        etas[t] = model.mu_q_eta(inp_t)
    return etas

def get_nearest_neighbors(word, vectors, vocab, num_words):
    index = vocab.index(word)
    query = vectors[index] 
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:num_words]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors


def get_centroid_vector(embeddings, vocab, keywords):
    # vectors of whole embeddings.
    indexes_keywords = [vocab.index(word) for word in keywords]
    queries = [embeddings[index] for index in indexes_keywords] # vectors of the keywords
    kmeans = cluster.KMeans(n_clusters=1, random_state=0).fit(queries)
    centroid = kmeans.cluster_centers_[0]
    print('centroid size: ', centroid.shape)
    return centroid

def main():
    if args.get_confidence:
        tokens, counts, times = load_data()
        indices = torch.split(torch.tensor(range(len(tokens))), 1000)
        
        
#     model_path = os.path.join('/home/yiyi/nlp_tm/', args.model_name)
    model_path = args.model_name
    with open(model_path, 'rb') as f:
        model = torch.load(f)
        
    model.to(device)
    
    with open(os.path.join(root_dir, 'preprocessed_data', 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = len(vocab)
    
    
    keywords = [ 'immigrant', 'immigrants',  'refugee', 'refugees','asylum',
      'migrant', 'migrants', 'internally displaced', 'UNHCR', 
      'asylees', 'Asylee',  'resettled', 'resettlements', 'asylee', 'pariah', 'pariahs', 're settlement', 'resettle', 'resettles',
      'immigration', 'Ateh', 'statelessness', 'Hagadera' , 'Domiz',
      'émigré', 'exile', 'displaced person', 'deserter' ]
#     keywords = ['immigrant', 'refugee']
    keywords = list(set([k.lower() for k in keywords]))

    keywords_selected = []
    for k in keywords: 
        if k in vocab:
            keywords_selected.append(k)
    print('keywords selected:', keywords_selected)
    
    neighbors_keywords_selected = []
    
    confidences_cos =[]
    confidences_eucl =[]
    model.eval()
    with torch.no_grad():

        embeddings = model.rho.weight
        embeddings= embeddings.cpu().numpy()
        
        
        #### top words similar to keywords
#         neighbors_keywords_selected = list(chain.from_iterable([get_nearest_neighbors(word, embeddings, vocab, 12)[1:11] for word in keywords_selected]))
#         neighbors_keywords_selected = list(set(neighbors_keywords_selected))
#         print('neighbors of keywords:', neighbors_keywords_selected, len(neighbors_keywords_selected))
#         indexes_keywords_neigh = [vocab.index(word) for word in neighbors_keywords_selected]
#         queries_kn = [embeddings[index] for index in indexes_keywords_neigh] 
        

        ### get centroid and the similar words of centroid and the vectors
        indexes_keywords = [vocab.index(word) for word in keywords_selected]
        queries = [embeddings[index] for index in indexes_keywords] 
        print(f'length of the keywords queries {len(queries)}')

        
        kmeans = cluster.KMeans(n_clusters=1, random_state=0)
        kmeans.fit(queries)
        X_dist = kmeans.transform(queries)**2
        square_distances = X_dist.sum(axis=1).round(2)
        sqrt_distances = np.sqrt(square_distances)
        mean_distances = np.mean(sqrt_distances)
        print('the distances of the centroid with the vectors:', mean_distances)
        centroid = kmeans.cluster_centers_[0]
        
#         centroid = np.mean(queries, axis=0)
#         ranks= embeddings.dot(centroid).squeeze()
#         denom = centroid.T.dot(centroid).squeeze()
#         denom = denom*np.sum(embeddings*2, 1)
#         denom = np.sqrt(denom)
#         ranks = ranks/denom
#         mostSimilar = []
#         [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
#         nearest_neighbors = mostSimilar[:10]
#         nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
#         print('the nearest neighbors of the centroid:', nearest_neighbors)
        
#         neigh_indices = [vocab.index(word) for word in nearest_neighbors]
#         neigh_q = [embeddings[index] for index in neigh_indices] 
#         print(f'length of the neighbors queries {len(neigh_q)}')
        
        queries.append(centroid)
        print(f'length of the neighbors queries {len(queries)}')
        n = queries
        # +neigh_q
        # +queries_kn
        n_arr = np.array(n)
        print(f"query shape {n_arr.shape}")
        
        ks =keywords_selected+['centroid']
        # +nearest_neighbors+neighbors_keywords_selected
        print(f"keywords {len(ks)}")
        
        tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=7)
        T = tsne.fit_transform(n_arr)
        
        colors = ['blue' for x in range(0,len(keywords_selected))]+['red']
        # ['orange' for x in range(0,len(nearest_neighbors))]
        # +['green' for x in range(0,len(neighbors_keywords_selected))]
        
        plt.figure(figsize=(40,10))
        plt.scatter(T[:, 0], T[:, 1], s=40, c=colors)

        for label, x, y in zip(ks, T[:, 0], T[:, 1]):
            plt.annotate(label, xy=(x+2, y+2), xytext=(0, 0), textcoords='offset points', fontsize=30)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.savefig(f'k_{args.num_topics}/dist_kn_{args.num_topics}.png')
        
        if args.get_confidence:
        
            rnn_inp = data.get_rnn_input(tokens, counts, times, args.num_times, vocab_size, len(tokens))
            etas = _eta_helper(model, rnn_inp)

            for idx, ind in enumerate(indices):
                data_batch, times_batch = data.get_batch(
                                tokens, counts, ind, len(vocab),300, temporal=True, times=times)
                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums
                eta_td = etas[times_batch.type('torch.LongTensor')].to(device)
                inp = torch.cat([eta_td, normalized_data_batch], dim=1).to(device)
                q_theta = model.q_theta(inp) # 1000, 800
                lr = nn.Linear(q_theta.shape[1], 300).to(device)
                mu_theta = lr(q_theta).cpu().numpy()
                for i in mu_theta:
                    eucl = distance.euclidean(i, centroid)
                    confidences_eucl.append(eucl)
                    cos = cosine_similarity(i, centroid)
                    confidences_cos.append(cos)

            # mu_theta = model.mu_q_theta(q_theta) # 1000, 75
    if args.get_confidence:
    
        df = pd.read_csv(f'k_{args.num_topics}/df_detm_topics_k{args.num_topics}.csv', index_col=0)
        
        df['euclidean']= confidences_eucl
        df['cossim']= confidences_cos

        print(df['topic'].value_counts())
        print(f"length of topics {len(df['topic'].value_counts())}")
        df.to_csv( f'k_{args.num_topics}/df_detm_topics_k{args.num_topics}_confidence.csv')
    
    
if __name__ == '__main__':
    main()