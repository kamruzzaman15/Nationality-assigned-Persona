import pandas as pd
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import random
import re
from langchain_community.llms import LlamaCpp
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

template = """

Context: {context}

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["prompt", "query"],
    template=template
)

llm = LlamaCpp(
    model_path="your-mpdel-path",
    n_gpu_layers = -1,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    f16_kv=True, 
)

prompt_data = pd.read_csv('original_dataset.csv')
prompt_data['response'] = ''
framing_data_invalid_result = []
irrelevent_response = 0
count = 0
for col, data in prompt_data.iterrows():
    
    query = f" The answer of the question must be country name. Answer the question with one sigle country name. We don't need any kind of explanation/details summary for the answers. Just provide the response directly. Don't include 'Answer:' or 'Response:' in your response. Just only the country name. Please don't start any kind of unnecessary/irrelevant conversation. Please answer the questions to the point as I asked. Remeber there is only one single country name is the answer."
    
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
    prompt_data.to_csv('mistral_response.csv', index=False, encoding='utf-8')