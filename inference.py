import requests
import os
from typing import Dict, Optional

import pandas as pd

class InferenceAPI:
    def __init__(self, api_url: str, api_headers: dict, parameters:dict = None, template=None, timeout: int=300):
        self.api_url = api_url
        self.template = template
        self.headers = api_headers
        self.parameters = parameters
        self.timeout = timeout
        
        
    def get_response_from_query(self, input: Dict[str, str], template: Optional[str] = None, parameters: dict = None) -> str:

        template = template if template is not None else self.template
        formatted_input = template.format(**input)
        parameters = parameters if parameters is not None else self.parameters

        payload = {
            "inputs": formatted_input,
            "parameters": parameters
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout).json()
        print(response)
        return response[0]['generated_text']


if __name__ == "__main__":
    # 사용 예시
    ## for HF endpoint
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    API_URL = "https://dm8tozpsdzcqu6an.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {os.environ.get('HUGGINGFACEHUB_API_TOKEN')}",
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
    
    test_df = pd.read_csv('./database/datasets/test_data_with_retrieve.csv', index_col=0)
    query = test_df.iloc[45, 0]

    hf_inference = InferenceAPI(API_URL, headers, parameters, template=template)

    input_data = {"query": query}

    response = hf_inference.get_response_from_query(input_data)
    print("Response:", response)
