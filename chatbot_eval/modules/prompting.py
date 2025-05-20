import re
from modules.openai_client import prompt_model

def linearize_demonstrations(demonstrations):
    """
    Converts a list of demonstration examples into a formatted text string.
    """
    prompt_text = ""
    for demo in demonstrations:
        prompt_text += f"Conversation:\n {demo.text}\n"
        prompt_text += f"Pass/Fail: {demo.pass_fail}\n"
        hashtags = ', '.join(demo.expected) if demo.expected else None
        prompt_text += f"Hashtags: {hashtags}\n\n"
    print(prompt_text)
    return prompt_text

def construct_prompt(demonstrations_text, conversation, tag_list):
    """
    Constructs the full prompt for the model.
    
    Parameters:
        demonstrations_text (str): The formatted text of the demonstrations.
        conversation (str): The text of the conversation to classify.
        tag_list (list): The full list of valid hashtags for prediction.
    
    Returns:
        str: The constructed prompt as a string.
    """
    bulletpoint_tag_list = "\n".join([f"- {tag}" for tag in tag_list])
    prompt = (
        f"Below are conversations and the hashtags that describe them:\n{demonstrations_text}"
        "Select the hashtags from the options below that are most applicable to the conversation. "
        "Return only the selected hashtags separated by commas (no additional text).\n"
        "Return 'None' if no hashtags are applicable.\n"
        f"Conversation: {conversation}\n"
        f"Options:\n{bulletpoint_tag_list}\n"
    )
    print(prompt)
    return prompt

def process_example(idx, test_data, demos_text, tags):
    """
    Run one test example through the model.
    Returns (result_dict, None) on success, (None, error_dict) on failure.
    """
    prompt = construct_prompt(demos_text, test_data.text, tags)
    try:
        response = prompt_model(prompt)
        print(f"{idx}: {response}")
        predicted = extract_valid_hashtags(response, tags)
        return {
            "review":      test_data.text,
            "true_labels": test_data.expected,
            "predicted":   predicted,
        }, None
    except Exception as e:
        return None, {
            "review":      test_data.text,
            "true_labels": test_data.expected,
            "error":       str(e),
        }
    
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