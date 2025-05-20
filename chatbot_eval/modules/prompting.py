import re
from modules.openai_client import prompt_model

def linearize_demonstrations_pass_fail(demonstrations):
    """
    Converts a list of demonstration examples into a formatted text string
    for the pass/fail classification step.
    """
    prompt_text = ""
    for demo in demonstrations:
        # Only include pass/fail status in the prompt
        prompt_text += f"Conversation:\n {demo.text}\n"
        prompt_text += f"Pass/Fail: {demo.pass_fail}\n\n"
    return prompt_text

def linearize_demonstrations_tagging(demonstrations):
    """
    Converts a list of demonstration examples into a formatted text string
    for the tagging step (only for failed conversations).
    """
    prompt_text = ""
    for demo in demonstrations:
        if demo.pass_fail == "Fail":  # Only show failed conversations with tags
            prompt_text += f"Conversation:\n {demo.text}\n"
            hashtags = ', '.join(demo.expected) if demo.expected else "None"
            prompt_text += f"Hashtags: {hashtags}\n\n"
    return prompt_text

def construct_prompt_pass_fail(demonstrations_text, conversation):
    """
    Constructs the prompt to classify whether the conversation passes or fails.
    
    Parameters:
        demonstrations_text (str): The formatted text of the demonstrations.
        conversation (str): The text of the conversation to classify.
    
    Returns:
        str: The constructed prompt for pass/fail classification.
    """
    prompt = (
        f"Below are conversations with their 'Pass/Fail' status:\n{demonstrations_text}\n"
        "Does the conversation pass or fail? Respond with 'Pass' or 'Fail'.\n"
        f"Conversation: {conversation}\n"
    )
    # print(prompt)
    return prompt

def construct_prompt_tagging(demonstrations_text, conversation, tag_list):
    """
    Constructs the prompt to select hashtags for a failed conversation.
    
    Parameters:
        demonstrations_text (str): The formatted text of the demonstrations.
        conversation (str): The text of the conversation to classify.
        tag_list (list): The full list of valid hashtags for prediction.
    
    Returns:
        str: The constructed prompt for selecting hashtags.
    """
    bulletpoint_tag_list = "\n".join([f"- {tag}" for tag in tag_list])
    prompt = (
        f"Below are conversations and the hashtags that describe them:\n{demonstrations_text}\n"
        "Select the hashtags from the options below that are most applicable to the conversation. "
        "Return only the selected hashtags separated by commas (no additional text).\n"
        f"Conversation: {conversation}\n"
        f"Options:\n{bulletpoint_tag_list}\n"
    )
    # print(prompt)
    return prompt

def process_example(idx, test_data, pass_fail_demos_text, tagging_demos_text, tags):
    """
    Run one test example through the model in two steps:
    Step 1: Classify pass/fail.
    Step 2: If fail, predict failure hashtags.
    
    Returns (result_dict, None) on success, (None, error_dict) on failure.
    """
    # Step 1: Check if the conversation passes or fails
    pass_fail_prompt = construct_prompt_pass_fail(pass_fail_demos_text, test_data.text)
    try:
        # First API call: pass/fail classification
        pass_fail_response = prompt_model(pass_fail_prompt)
        print(f"{idx}: {pass_fail_response}")
        
        if "Pass" in pass_fail_response:
            predicted_pass_fail = "Pass"
            predicted_tags = []
        elif "Fail" in pass_fail_response:
            predicted_pass_fail = "Fail"
            
            # Step 2: If it failed, predict hashtags (tags)
            tagging_prompt = construct_prompt_tagging(tagging_demos_text, test_data.text, tags)
            # Second API call: tag prediction
            tagging_response = prompt_model(tagging_prompt)
            print(f"Tags: {tagging_response}")
            
            predicted_tags = extract_valid_hashtags(tagging_response, tags)
        else:
            predicted_pass_fail = "Pass*"
            predicted_tags = []

        return {
            "review": test_data.text,
            "pass_fail": test_data.pass_fail,
            "tags": test_data.expected,
            "predicted_pass_fail": predicted_pass_fail,
            "predicted_tags": predicted_tags,
        }, None
    
    except Exception as e:
        return None, {
            "review": test_data.text,
            "true_labels": test_data.expected,
            "error": str(e),
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