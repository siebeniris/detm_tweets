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
from collections import Counter

import data 

from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import functional as F

from detm import DETM
from utils import nearest_neighbors, get_topic_coherence

import pandas as pd
import csv

parser = argparse.ArgumentParser(description='Infer topics from the pre-trained embedded topic model....')

### data and file related arguments
parser.add_argument('--model_name', type=str, default='results/detm_twitter_K_50_Htheta_800_Optim_adam_Clip_0.0_ThetaAct_relu_Lr_0.005_Bsz_1000_RhoSize_300_L_3_minDF_10_trainEmbeddings_1_val_ppl_2638.9_epoch_20', help='Path of the pre-trained model...')
parser.add_argument('--num_times', type=int, default=8, help='number of years')
parser.add_argument('--num_topics', type=int, default=50, help='number of years')



args = parser.parse_args()
root_dir = '/home/yiyi/nlp_tm/'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")



with open(os.path.join(root_dir, 'preprocessed_data', 'vocab.pkl'), 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

print('loading the vocab ..', vocab_size)

# model_path = os.path.join(root_dir, args.model_name)
model_path = args.model_name
print('loading the model:', model_path)
with open(model_path, 'rb') as f:
    model=torch.load(f)

model.to(device)

output_dir = f'k_{args.num_topics}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def load_data(input_dir= os.path.join(root_dir, 'preprocessed_data')):
    token_file = os.path.join(input_dir, 'bow_tokens')
    count_file = os.path.join(input_dir, 'bow_counts')
    time_file = os.path.join(input_dir, 'bow_timestamps')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    times = scipy.io.loadmat(time_file)['timestamps'].squeeze()
    
    return tokens, counts, times


def _eta_helper(rnn_inp):
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

def get_theta(eta, bows):
    """
    Document proportions....
    """
    model.eval()
    with torch.no_grad():
        inp = torch.cat([bows, eta], dim=1)
        q_theta = model.q_theta(inp)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta 


def main():
#     csvfile = open(f'k_{args.num_topics}/topic_visualization_topwords.csv', 'w', newline='')
#     csvwriter = csv.writer(csvfile, delimiter= ',', quoting=csv.QUOTE_MINIMAL)
#     csvwriter.writerow(['Topic', 'Time', 'Topic Words'])
    
    
#     tokens, counts, times = load_data()
#     print('documents nr: ', len(tokens))
#     indices = torch.split(torch.tensor(range(len(tokens))), 1000)
    
    print('starting evaluating....')
    theta_weights = []
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha.to(device)
#         rnn_inp = data.get_rnn_input(tokens, counts, times, args.num_times, vocab_size, len(tokens))
#         etas = _eta_helper(rnn_inp)
        scipy.io.savemat(f'k_{args.num_topics}/'+'alpha.mat', {'values': alpha}, do_compression=True)
        
        rho = model.rho.weight.cpu().numpy()
        scipy.io.savemat(f'k_{args.num_topics}/rho.mat', {'values':rho}, do_compression=True)
        
        ### get topics beta.
        beta = model.get_beta(alpha).to(device)
        beta = beta.cpu().numpy()
        scipy.io.savemat(f'k_{args.num_topics}/'+'beta.mat', {'values': beta}, do_compression=True)
        print('beta: ', beta.size())
#         print('\n')
#         print('#'*100)
#         print('Visualize topics...')
        
#         timeslist = [x for x in range(8)]
#         topics_words = []
#         for k in range(args.num_topics):
#             for t in timeslist:
#                 gamma = beta[k, t, :]
#                 top_words = list(gamma.cpu().numpy().argsort()[-20:][::-1])
#                 topic_words = [vocab[a] for a in top_words]
#                 topics_words.append(' '.join(topic_words))
                
#                 csvwriter.writerow([k, t+2013, topic_words])
                
#                 print('Topic {} .. Time: {} ===> {}'.format(k, t+2013, topic_words)) 
        
#         ### get whole data############################
#         tokens, counts, times = load_data()
#         print('documents nr: ', len(tokens))
#         indices = torch.split(torch.tensor(range(len(tokens))), 1000)
        
#         ### infer topics for each tweet ##########
#         for idx, ind in enumerate(indices):
#             data_batch, times_batch = data.get_batch(
#                             tokens, counts, ind, len(vocab),300, temporal=True, times=times)
#             sums = data_batch.sum(1).unsqueeze(1)
#             normalized_data_batch = data_batch / sums
#             eta_td = etas[times_batch.type('torch.LongTensor')]
            
#             ############ get topic words#####
#             alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :].to(device)            
#             theta = get_theta(eta_td, normalized_data_batch).to(device)
#             theta = theta.cpu().numpy()
#             theta_weights.append(theta)
            
#     print('collected all theta weights...')
#     theta_weights_list = []
#     for w in theta_weights:
#         for i in w:
#             theta_weights_list.append(i)
#     theta_weights_arr = np.stack(theta_weights_list, axis=0)
#     print("theta shape: ", theta_weights_arr.shape)
    
#     inds = np.argmax(theta_weights_arr, axis=1)

#     df = pd.read_csv('/home/yiyi/nlp_tm/datasets/df_detm_ids.csv', index_col=0)
#     df['topic']= inds
#     df.to_csv(f'k_{args.num_topics}/df_detm_topics_k{args.num_topics}.csv')
#     print(sorted(Counter(inds).items()))
    
          
    
if __name__ == '__main__':
    main()