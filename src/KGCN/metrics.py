import numpy as np

def dcg_at_k(r, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is binary (nonzero is relevant).
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return dcg

def ndcg_at_k(r, k):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is binary (nonzero is relevant).
    Returns:
        Normalized discounted cumulative gain
    """
    assert k >= 1
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def recall_at_k(r, k, all_pos_num):
    """Score is recall @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Recall @ k
    """
    assert k >= 1
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num