import pandas as pd
import ollama
client = ollama.Client()

prompts_data = pd.read_csv('original_dataset.csv')

prompts_data_result = []

for _, data in prompts_data.iterrows():
    prompt = data['prompt']

    try:
        gemma_response_list = client.chat(model="gemma", messages=[{"role": "user", "content": prompt}])
        if not gemma_response_list:
            print(f"No response received for prompt: {prompt}")
            data['response'] = "No Response"
        else:
            gemma_response = gemma_response_list[0].content.strip() 
            data['response'] = gemma_response

        prompts_data_result.append(data)

    except Exception as e:
        print('Exception occurred: ', e)
        data['response'] = gemma_response_list
        prompts_data_result.append(data)

df = pd.DataFrame(prompts_data_result)
df.to_csv('gemma_response.csv', index=False, encoding='utf-8')