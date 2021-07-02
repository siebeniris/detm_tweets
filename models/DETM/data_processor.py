import os
import csv
import pickle
import random
import itertools
import string
import wsgiref.validate
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat, loadmat
from typing import Dict, List

rootdir = Path(__file__).parent.parent.parent

print('root directory: ', rootdir)  # .../nlp_tm

# Maximum / minimum document frequency
max_df = 0.7
min_df = 10  # choose desired value for min_df


def load_stopwords(file: str = os.path.join(rootdir, 'models/DETM/scripts/stops.txt')) -> List:
    """
    Load stopwords.
    """
    with open(file) as f:
        stops = f.read().split('\n')
        return stops


def load_data(file: str = 'datasets/un-general-debates.csv', flag_split_by_paragraph: bool = True,
              timestamp_filed: str = 'year', docs_field: str = 'text') -> (List, List):
    """
    wehter to split the documents by paragraph or not.
    """

    df = pd.read_csv(file)
    all_timestamps_ini = df[timestamp_filed].tolist()
    all_docs_ini = df[docs_field].tolist()
    print('size of df:', len(df))

    if flag_split_by_paragraph:
        print('splitting by paragraphs...')
        docs = []
        timestamps = []
        for dd, doc in enumerate(all_docs_ini):
            splitted_doc = doc.split('.\n')
            for ii in splitted_doc:
                docs.append(ii)
                timestamps.append(all_timestamps_ini[dd])
        return docs, timestamps
    else:
        docs = all_docs_ini
        timestamps = all_timestamps_ini
        return docs, timestamps


def text_processing(docs: List, output_dir: str = os.path.join(rootdir, 'preprocessed_data'),
                    outputfile='un-general-debates-all-docs-splitparagraphs') -> List:
    """
    Preprocessing the texts: remove punctuations, remove the texts whose length are less than 2
    """
    # remove punctuations and the numbers.
    # docs = [[w.lower().replace("’", " ").replace("'", " ").replace('\ufeff', '').translate(
    #     str.maketrans('', '', string.punctuation + "0123456789")) for w in docs[doc].split()] for doc in
    #     range(len(docs))]
    # docs = [[w for w in docs[doc] if len(w) > 1] for doc in range(len(docs))] # already done.
    docs = [" ".join(docs[doc]) for doc in range(len(docs))]

    print('size of docs:', len(docs))
    ### save the raw text.
    with open(os.path.join(output_dir, outputfile), 'w') as writer:
        for line in docs:
            writer.write(line + '\n')

    return docs


def remove_empty(in_docs, in_timestamps):
    """
    Remove the empty doc.
    """
    out_docs = []
    out_timestamps = []
    for ii, doc in enumerate(in_docs):
        if (doc != []):
            out_docs.append(doc)
            out_timestamps.append(in_timestamps[ii])
    return out_docs, out_timestamps


def remove_by_threshold(in_docs, in_timestamps, thr):
    """
    Remove the docs with threshold.
    """
    out_docs = []
    out_timestamps = []
    for ii, doc in enumerate(in_docs):
        if (len(doc) > thr):
            out_docs.append(doc)
            out_timestamps.append(in_timestamps[ii])
    return out_docs, out_timestamps


def split_data(docs: List, timestamps: List, stops: List, min_df: int = min_df, max_df: float = 0.7,
               output_dir: str = os.path.join(rootdir, 'preprocessed_data')):
    """
    Create count vectorizer.
    Split data into train, test and val datasets.
    """
    print('counting document frequency of words... ')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
    cvz = cvectorizer.fit_transform(docs).sign()

    print('building the vocabulary...')
    sum_counts = cvz.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0, v]
    word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
    id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
    print('  initial vocabulary size: {}'.format(v_size))

    # sort elements in vocabulary    
    idx_sort = np.argsort(sum_counts_np)
    vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]

    # Filter out stopwords (if any)
    vocab_aux = [w for w in vocab_aux if w not in stops]
    print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))

    # Create dictionary and inverse dictionary
    vocab = vocab_aux
    del vocab_aux
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])

    # Create mapping of timestamps
    all_times = sorted(set(timestamps))
    time2id = dict([(t, i) for i, t in enumerate(all_times)])
    id2time = dict([(i, t) for i, t in enumerate(all_times)])
    time_list = [id2time[i] for i in range(len(all_times))]

    #  Split in train/test/valid
    print('tokenizing documents and splitting into train/test/valid...')
    num_docs = cvz.shape[0]
    trSize = int(np.floor(0.85 * num_docs))
    tsSize = int(np.floor(0.10 * num_docs))
    vaSize = int(num_docs - trSize - tsSize)
    del cvz
    idx_permute = np.random.permutation(num_docs).astype(int)

    #  Remove words not in train_data
    vocab = list(set([w for idx_d in range(trSize) for w in docs[idx_permute[idx_d]].split() if w in word2id]))
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])
    print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

    docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
    timestamps_tr = [time2id[timestamps[idx_permute[idx_d]]] for idx_d in range(trSize)]
    docs_ts = [[word2id[w] for w in docs[idx_permute[idx_d + trSize]].split() if w in word2id] for idx_d in
               range(tsSize)]
    timestamps_ts = [time2id[timestamps[idx_permute[idx_d + trSize]]] for idx_d in range(tsSize)]
    docs_va = [[word2id[w] for w in docs[idx_permute[idx_d + trSize + tsSize]].split() if w in word2id] for idx_d in
               range(vaSize)]
    timestamps_va = [time2id[timestamps[idx_permute[idx_d + trSize + tsSize]]] for idx_d in range(vaSize)]
    del docs

    print('  number of documents (train): {} [this should be equal to {} and {}]'.format(len(docs_tr), trSize,
                                                                                         len(timestamps_tr)))
    print('  number of documents (test): {} [this should be equal to {} and {}]'.format(len(docs_ts), tsSize,
                                                                                        len(timestamps_ts)))
    print('  number of documents (valid): {} [this should be equal to {} and {}]'.format(len(docs_va), vaSize,
                                                                                         len(timestamps_va)))

    docs_tr, timestamps_tr = remove_empty(docs_tr, timestamps_tr)
    docs_ts, timestamps_ts = remove_empty(docs_ts, timestamps_ts)
    docs_va, timestamps_va = remove_empty(docs_va, timestamps_va)

    # Remove test documents with length=1
    docs_ts, timestamps_ts = remove_by_threshold(docs_ts, timestamps_ts, 1)

    # write the vocabulary and timestamps
    with open(os.path.join(output_dir, 'vocab.txt'), "w") as f:
        for v in vocab:
            f.write(v + '\n')

    with open(os.path.join(output_dir, 'timestamps.txt'), "w") as f:
        for v in timestamps:
            f.write(str(v) + '\n')

    with open(os.path.join(output_dir, 'vocab.pkl'), "wb") as f:
        pickle.dump(vocab, f)

    with open(os.path.join(output_dir, 'timestamps.pkl'), "wb") as f:
        pickle.dump(time_list, f)

    return docs_tr, docs_ts, docs_va, timestamps_tr, timestamps_ts, timestamps_va, len(vocab)


############# util functions ############################
def create_list_words(in_docs):
    return [x for y in in_docs for x in y]


def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]


def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()


def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc, :].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc, :].data] for doc in range(n_docs)]
    return indices, counts


#########################################################

def get_data(docs_tr: List, docs_ts: List, docs_va: List, timestamps_tr: List,
             timestamps_ts: List, timestamps_va: List, len_vocab: int,
             output_dir: str = os.path.join(rootdir, 'preprocessed_data')):
    # Split test set in 2 halves
    print('splitting test documents in 2 halves...')
    docs_ts_h1 = [[w for i, w in enumerate(doc) if i <= len(doc) / 2.0 - 1] for doc in docs_ts]
    docs_ts_h2 = [[w for i, w in enumerate(doc) if i > len(doc) / 2.0 - 1] for doc in docs_ts]

    # Getting lists of words and doc_indices
    print('creating lists of words...')

    words_tr = create_list_words(docs_tr)
    words_ts = create_list_words(docs_ts)
    words_ts_h1 = create_list_words(docs_ts_h1)
    words_ts_h2 = create_list_words(docs_ts_h2)
    words_va = create_list_words(docs_va)

    print('  len(words_tr): ', len(words_tr))
    print('  len(words_ts): ', len(words_ts))
    print('  len(words_ts_h1): ', len(words_ts_h1))
    print('  len(words_ts_h2): ', len(words_ts_h2))
    print('  len(words_va): ', len(words_va))

    # Get doc indices
    print('getting doc indices...')

    doc_indices_tr = create_doc_indices(docs_tr)
    doc_indices_ts = create_doc_indices(docs_ts)
    doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
    doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
    doc_indices_va = create_doc_indices(docs_va)

    print(
        '  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
    print(
        '  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))
    print('  len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h1)),
                                                                               len(docs_ts_h1)))
    print('  len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h2)),
                                                                               len(docs_ts_h2)))
    print(
        '  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_va)))

    # Number of documents in each set
    n_docs_tr = len(docs_tr)
    n_docs_ts = len(docs_ts)
    n_docs_ts_h1 = len(docs_ts_h1)
    n_docs_ts_h2 = len(docs_ts_h2)
    n_docs_va = len(docs_va)

    # Create bow representation
    print('creating bow representation...')

    bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len_vocab)
    bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len_vocab)
    bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len_vocab)
    bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len_vocab)
    bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len_vocab)

    savemat(os.path.join(output_dir, 'bow_tr_timestamps'), {'timestamps': timestamps_tr}, do_compression=True)
    savemat(os.path.join(output_dir, 'bow_ts_timestamps'), {'timestamps': timestamps_ts}, do_compression=True)
    savemat(os.path.join(output_dir, 'bow_va_timestamps'), {'timestamps': timestamps_va}, do_compression=True)

    # Split bow into token/value pairs
    print('splitting bow into token/value pairs and saving to disk...')

    bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
    savemat(os.path.join(output_dir, 'bow_tr_tokens'), {'tokens': bow_tr_tokens}, do_compression=True)
    savemat(os.path.join(output_dir, 'bow_tr_counts'), {'counts': bow_tr_counts}, do_compression=True)

    bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
    savemat(os.path.join(output_dir, 'bow_ts_tokens'), {'tokens': bow_ts_tokens}, do_compression=True)
    savemat(os.path.join(output_dir, 'bow_ts_counts'), {'counts': bow_ts_counts}, do_compression=True)

    bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
    savemat(os.path.join(output_dir, 'bow_ts_h1_tokens'), {'tokens': bow_ts_h1_tokens}, do_compression=True)
    savemat(os.path.join(output_dir, 'bow_ts_h1_counts'), {'counts': bow_ts_h1_counts}, do_compression=True)

    bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
    savemat(os.path.join(output_dir, 'bow_ts_h1_tokens'), {'tokens': bow_ts_h2_tokens}, do_compression=True)
    savemat(os.path.join(output_dir, 'bow_ts_h1_counts'), {'counts': bow_ts_h2_counts}, do_compression=True)

    bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
    savemat(os.path.join(output_dir, 'bow_va_tokens'), {'tokens': bow_va_tokens}, do_compression=True)
    savemat(os.path.join(output_dir, 'bow_va_counts'), {'counts': bow_va_counts}, do_compression=True)

    print('Data ready !!')
    print('*************')


def main(file: str = 'datasets/un-general-debates.csv', output_dir: str = os.path.join(rootdir, 'preprocessed_data')):
    stops = load_stopwords()
    print('loading data ....')
    docs, timestamps = load_data(file=file)
    print('text preprocessing...')
    docs = text_processing(docs)
    print(docs[:10])

    docs_tr, docs_ts, docs_va, timestamps_tr, timestamps_ts, timestamps_va, len_vocab = split_data(docs, timestamps,
                                                                                                   stops)
    get_data(docs_tr, docs_ts, docs_va, timestamps_tr, timestamps_ts, timestamps_va, len_vocab, output_dir=output_dir)


if __name__ == '__main__':
    main()
