import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from scipy.special import rel_entr
from tqdm import tqdm 


_DISTANCE_METRICS = ['cosine', 'euclidean']

def _check_coreset_size(x, n, indices_unlabeled):
    if n > x.shape[0]:
        raise ValueError(f'n (n={n}) is greater the number of available samples (num_samples={x.shape[0]})')
    
    if len(indices_unlabeled) < n:
        raise ValueError(f'n (n={n}) is greater the number of unlabel samples (num_samples={len(indices_unlabeled)})')


def _cosine_distance(a, b, normalized=False):
    sim = np.matmul(a, b.T)
    if not normalized:
        sim = sim / np.dot(np.linalg.norm(a, axis=1)[:, np.newaxis],
                           np.linalg.norm(b, axis=1)[np.newaxis, :])
    return np.arccos(sim) / np.pi


def _euclidean_distance(a, b, normalized=False):
    _ = normalized
    return pairwise_distances(a, b, metric='euclidean')


def greedy_coreset(x, indices_unlabeled, indices_labeled, n, distance_metric='cosine',
                   batch_size=100, normalized=False):

    _check_coreset_size(x, n, indices_unlabeled)

    num_batches = int(np.ceil(x.shape[0] / batch_size))
    ind_new = []

    if distance_metric == 'cosine':
        dist_func = _cosine_distance
    elif distance_metric == 'euclidean':
        dist_func = _euclidean_distance
    else:
        raise ValueError(f'Invalid distance metric: {distance_metric}. '
                         f'Possible values: {_DISTANCE_METRICS}')

    for _ in range(n):
        indices_s = np.concatenate([indices_labeled, ind_new])
        dists = np.array([], dtype=np.float32)
        for batch in np.array_split(x.iloc[indices_unlabeled], num_batches, axis=0):

            dist = dist_func(batch, x.iloc[indices_s], normalized=normalized)

            sims_batch = np.amin(dist, axis=1)
            dists = np.append(dists, sims_batch)

        dists[ind_new] = -np.inf
        index_new = np.argmax(dists)

        ind_new.append(index_new)
    return np.array(ind_new)


def query_by_contrastive_active_learning(df, args, batch_size=100, k=10):

    nn = NearestNeighbors(n_neighbors=10, algorithm="kd_tree")
    embeddings = pd.DataFrame(df['cls_embeddings'].to_list())
    nn.fit(embeddings)
    scores = []
    probas = df['probabilities']

    num_batches = int(np.ceil(len(df) / batch_size))
    offset = 0
    for batch_idx in tqdm(np.array_split(np.arange(df.index.shape[0]), num_batches,
                                    axis=0)):

        nn_indices = nn.kneighbors(embeddings.iloc[batch_idx],
                                    n_neighbors=k + 1, # the nearest neighbor is itself which is useless info.
                                    return_distance=False)

        kl_divs = list(map(lambda v: np.mean([
                    sum(rel_entr(probas.iloc[i], probas.iloc[v])) for i in nn_indices[v - offset][1:]
                ]), 
            batch_idx))

        scores.extend(kl_divs)
        offset += batch_idx.shape[0]

    scores = np.array(scores)
    seleted_index = np.argpartition(-scores, args.al_config["query_size"])[:args.al_config["query_size"]]

    return seleted_index

def query_active_learning_data(df, args, labeled_docid=[]):
    # Calulate active learning score of unlabeled data
    if args.al_config["query_method"] == "prediction_entropy":
        # Entropy of the prediction probability, select the highest
        score = [entropy(item) for item in df["probabilities"]]

    elif args.al_config["query_method"] == "least_confidence":
        # Probability of the most likely class, select the lowest
        score = [-max(item) for item in df["probabilities"]]

    elif args.al_config["query_method"] == "breaking_ties":
        # Difference between the probability of the 2 most likely class, select the lowest
        score = list(map(lambda probas: -(probas[0] - probas[1]), [sorted(item, reverse=True)[:2] for item in df["probabilities"]]))

    elif args.al_config["query_method"] == "coreset":
        unlabel_index = df.reset_index().index[~df['docid'].isin(labeled_docid)]
        labeled_index = df.reset_index().index[df['docid'].isin(labeled_docid)]
        selected_index = greedy_coreset(pd.DataFrame(df['cls_embeddings'].to_list()), unlabel_index, labeled_index, args.al_config["query_size"])
        score = [1 if ind in set(selected_index) else 0 for ind in range(len(df))]

    elif args.al_config["query_method"] == "cal":
        # KL divergence of the prediciton probability of N (set as 10 currently) nearest neighbours 
        # clustered by the BERT <CLS> embeddings, select the highest
        selected_index = query_by_contrastive_active_learning(df, args=args)
        score = [1 if ind in set(selected_index) else 0 for ind in range(len(df))]

    elif args.al_config["query_method"] == "random_sample":
        # Random sampling
        score = [random.uniform(0, 1) for item in df["probabilities"]]

    else:
        raise ValueError(f"Active learning method - {args.al_config['query_method']} - not found/implemented.")

    df['active_learning_score'] = score

    # Select top N data
    query_data = df[~df['docid'].isin(set(labeled_docid))].sort_values("active_learning_score", ascending=False)[:args.al_config["query_size"]]
    return query_data
