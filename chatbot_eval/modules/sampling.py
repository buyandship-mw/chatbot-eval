import random

def sample_demonstrations(dataset, k=16):
    """
    Samples k random demonstrations from the dataset.
    """
    return random.sample(dataset, k)

def relevant_hashtags_last(demonstrations, query_hashtags):
    """
    Orders demonstrations so that those most relevant to the query hashtags are placed last.
    Use embeddings or other methods to determine relevance.
    """
    pass