import requests
import os

import pandas as pd

from inference import InferenceAPI


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

template = """다음은 내가 처한 상황이야. 변호사로서 내 상황에 적용할 수 있는 구체적인 법령과 그 이유를 설명해줘.
    {query}
    ---
    답변:
    """

inference_api = InferenceAPI(API_URL, headers, parameters, template=template, timeout=300)  # 5분 타임아웃 설정



df_list = ["/home/jongbin/AlphaMist_generate_answer/results/AlphaMist_baseline1_temp_result.csv"]

for i, df_path in enumerate(df_list):
    df = pd.read_csv(df_path, index_col=0)
    
    if '답변' not in df.columns:
        df['답변'] = None
    
    for idx in df.index:
        if pd.isna(df.at[idx, '답변']):
            try:
                print('send API call...')
                input_data = {'query': df.at[idx, '자연어 쿼리']}
                response = inference_api.get_response_from_query(input_data)
                df.at[idx, '답변'] = response
                df.to_csv(df_path)  # 각 행마다 저장하여 진행 상태를 유지
                print('get response!')
            except requests.exceptions.Timeout:
                print(f"Timeout error processing row {idx} in {df_path}")
                continue
            except Exception as e:
                print(f"Error processing row {idx} in {df_path}: {e}")
                continue

print("Processing completed.")