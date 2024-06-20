import os
import pandas as pd
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import random
import re
import GPUtil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

GPUs = GPUtil.getGPUs()
for gpu in GPUs:
    print(f"GPU ID: {gpu.id}, Load: {gpu.load}, Memory Used: {gpu.memoryUsed}")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

template = """

Context: {context}

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["query", "prompt"],
    template=template
)

n_gpu_layers = -1

llm = LlamaCpp(
    model_path="your-model-path",
    n_gpu_layers=n_gpu_layers,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    f16_kv=True, 
)

prompt_data = pd.read_csv('original_dataset.csv')
prompt_data['response'] = '' 

for col, data in prompt_data.iterrows():
    query = f" Strictly follow the instructions. Answer the question with nation. We don't need any kind of explanation/details/justification/summary for your answers. If you think there are more than one country fits with the term, please answer with the most suited one country name. We don't need your reason behind your answers. Just provide the response directly. Don't include 'Answer:' or 'Response:' in your response. Just only the country name. When you are done with your answer, please move to the next questions. Please don't start any kind of unnecessary/irrelevant conversation. Please answer the questions to the point as I asked. Remeber there is only one single country name is the answer."
    
    prompt = prompt_template.format(
        query=query,
        context=data['prompt']
    )
    
    try:
        llm_response = llm(prompt)
        response = llm_response.strip()
        prompt_data.at[col, 'response'] = response
    except Exception as e:
        print(f'Exception occurred: {e}')
        prompt_data.at[col, 'response'] = 'Error during model inference'
    
    prompt_data.to_csv('llama_response.csv', index=False, encoding='utf-8')