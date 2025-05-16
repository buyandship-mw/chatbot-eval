import random

def sample_demonstrations(dataset, k=16):
    """
    Samples k random demonstrations from the dataset.
    """
    return random.sample(dataset, k)

def linearize_demonstrations(demonstrations):
    """
    Converts a list of demonstration examples into a formatted text string.
    """
    prompt_text = ""
    for demo in demonstrations:
        prompt_text += f"Review: {demo.text}\n"
        prompt_text += f"Hashtags: {', '.join(demo.expected)}\n\n"
    return prompt_text

def relevant_hashtags_last(demonstrations, query_hashtags):
    """
    Orders demonstrations so that those most relevant to the query hashtags are placed last.
    Use embeddings or other methods to determine relevance.
    """
    pass