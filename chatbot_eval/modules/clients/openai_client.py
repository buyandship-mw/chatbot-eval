import os
import configparser
from openai import OpenAI

# Read API key from config file
config = configparser.ConfigParser()
config_file = os.path.join(os.path.dirname(__file__), "../../config/config.ini")
config.read(config_file)
api_key = config['openai']['api_key']
model = config['openai']['model']

# Create a new OpenAI client
client = OpenAI(api_key=api_key)

def prompt_model(prompt, temp=1.0):
    """
    Sends a prompt to the OpenAI model and returns the response.
    Args:
        prompt (str): The prompt to send to the model.
        temp (float): The temperature for the model's response.
    Returns:
        str: The model's response.
    """
    completion = client.chat.completions.create(
        model=model,
        store=True,
        temperature=temp,
        messages=[
            {"role": "user", 'content': prompt}
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    # Test the LLM client is working
    text = prompt_model("Say hello world back to me.")
    print(text)