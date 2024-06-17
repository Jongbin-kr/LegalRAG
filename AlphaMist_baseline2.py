import requests

import pandas as pd

from inference import InferenceAPI

import os
os.environ['HUGGINGFACE_HUB_API_TOKEN'] = "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"
API_URL = "https://dm8tozpsdzcqu6an.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
    "Accept" : "application/json",
    "Authorization": f"Bearer {os.environ.get('HUGGINGFACE_HUB_API_TOKEN')}",
    "Content-Type": "application/json" 
}

parameters = {
        "max_new_tokens": 6000
}

template = """다음은 내가 처한 상황이야. 
{query}

다음은 나와 유사한 상황 판례야.
{docs}

변호사로서 내 상황에 적용할 수 있는 구체적인 법령과 그 이유를 설명해줘.
---
response:

"""

inference_api = InferenceAPI(API_URL, headers, parameters, template=template, timeout=240)  # 5분 타임아웃 설정

df_list = ["./df1.csv", "./df2.csv", "./df3.csv", "./df4.csv"]

for i, df_path in enumerate(df_list):
    df = pd.read_csv(df_path, index_col=0)
    
    if 'response' not in df.columns:
        df['response'] = None
    
    for idx in df.index:
        if pd.isna(df.at[idx, 'response']):
            try:
                print('send API call...')
                input_data = {'query': df.at[idx, 'query'], 'docs': df.at[idx, 'fact']}
                response = inference_api.get_response_from_query(input_data)
                df.at[idx, 'response'] = response
                df.to_csv(df_path)  # 각 행마다 저장하여 진행 상태를 유지
                print('get response!')
            except requests.exceptions.Timeout:
                print(f"Timeout error processing row {idx} in {df_path}")
                continue
            except Exception as e:
                print(f"Error processing row {idx} in {df_path}: {e}")
                continue

print("Processing completed.")