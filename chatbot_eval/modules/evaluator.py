import re

def construct_prompt(demonstrations_text, review, tag_list):
    """
    Constructs the full prompt for the model.
    
    Parameters:
        demonstrations_text (str): The formatted text of the demonstrations.
        review (str): The text of the review to classify.
        tag_list (list): The full list of valid hashtags for prediction.
    
    Returns:
        str: The constructed prompt as a string.
    """
    bulletpoint_tag_list = "\n".join([f"- {tag}" for tag in tag_list])
    prompt = f"Below are conversations and the hashtags that describe them:\n{demonstrations_text}"
    prompt += "Select the hashtags from the options below most applicable to the conversation. Return only the selected hashtags separated by commas.\n"
    prompt += f"Review: {review}\n"
    prompt += f"Options:\n{bulletpoint_tag_list}"
    return prompt

def extract_valid_hashtags(response, tag_list):
    """
    Extracts valid hashtags from the model's response.
    
    Parameters:
        response (str): The raw response string from the model.
        tag_list (list): The list of valid hashtags.
    
    Returns:
        list: A list of valid hashtags extracted from the response.
    """
    # Normalize the response by removing extra spaces and splitting into lines
    response = response.strip()
    
    # Use a regular expression to extract all words starting with '#'
    extracted_tags = re.findall(r"#\w+", response)
    
    # Filter the extracted tags to include only those in the valid tag list
    valid_tags = [tag for tag in extracted_tags if tag in tag_list]
    
    return valid_tags