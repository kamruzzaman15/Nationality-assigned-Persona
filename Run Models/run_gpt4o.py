import openai
import pandas as pd
import time 

person_data = pd.read_csv('original_dataset.csv')
openai.api_key = 'your-openai-key'

def chatGPTResponse(message=''):
    conversation = []
    conversation.append({'role': 'system', 'content': 'You are a helpful assistant.'})
    conversation.append({'role': 'user', 'content': message})
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=conversation,
        request_timeout=10
    )

    result = response.choices[0].message.content
    return result

for index, row in person_data.iterrows():
    prompt = row['prompt']
    
    try:
        response = chatGPTResponse(prompt)
        person_data.at[index, 'response'] = response.strip()
    except Exception as e:
        print(f"An exception occurred for index {index}: {e}")
        time.sleep(60)
        person_data.at[index, 'response'] = "Error generating response."

person_data.to_csv('gpt4o_response.csv', index=False, encoding='utf-8')