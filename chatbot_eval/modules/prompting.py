def linearize_demonstrations(demonstrations):
    """
    Converts a list of demonstration examples into a formatted text string.
    """
    prompt_text = ""
    for demo in demonstrations:
        prompt_text += f"Conversation:\n {demo.text}\n"
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