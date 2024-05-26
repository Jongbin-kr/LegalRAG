import os
from typing import *

from openai import OpenAI
import pandas as pd



class GPTQueryMaker():
    def __init__(self, prompt, model="gpt-3.5-turbo"):
        self.prompt = prompt
        self.model = model
        
        
    def generate_simple_query(self, crime_facts: str) -> str:
        """
        주어진 범죄사실을 일반인이 이해할 수 있는 자연어 쿼리로 변환합니다.
        
        Args:
        crime_facts (str): 복잡한 범죄사실을 설명하는 텍스트. 
        문자열 포매팅의 형태로 prompt에 
        
        Returns:
        str: 간단한 자연어 쿼리
        """
        
        formatted_prompt = self.prompt.format(crime_facts)
        
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system", "content": self.prompt
                }
            ],
            model=self.model,
        )

        
        return response.choices[0].message.content
    
    
    def generate_querys_from_dataframe(self, df: pd.DataFrame, new_col_name: str="자연어 쿼리", from_col_name: str="범죄 사실") -> pd.DataFrame:
        df[new_col_name] = df[from_col_name].apply(self.generate_querys_from_dataframe)
        
        return df
     
    
if __name__ == "__main__":
    df = pd.read_csv("./datasets/dataset.csv", index_col="index")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 데이터프레임 크기를 구합니다
    total_size = len(df)

    # 60%, 20%, 20% 크기를 구합니다
    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)
    test_size = total_size - train_size - val_size

    # 데이터를 나눕니다
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]

    # 각각의 데이터프레임을 CSV 파일로 저장합니다
    train_data.to_csv('./datasets/train_data.csv', index=False)
    val_data.to_csv('./datasets/val_data.csv', index=False)
    test_data.to_csv('./datasets/test_data.csv', index=False)
    print(len(train_data), len(val_data), len(test_data))

    # 테스트 데이터에 대하여 다음과 같은 프롬프트를 활용해 자연어 쿼리를 만듭니다.
    prompt = """당신은 복잡한 범죄 사실들을 마치 일반인이 얘기하듯 자연스러운 문장으로 풀어쓰는 한국어 챗봇입니다.
    제가 범죄사실을 말하면, 당신은 실제 변호사에게 물어보듯 해당 범죄사실을 자연스러운 문장으로 풀어쓰세요. 
    이때 다음과 같은 규칙들을 반드시 준수해야합니다.
    1. 당신이 범죄 사실을 풀어쓸 때 담당할 역할을 피의자, 피해자, 혹은 그 주변인물 중 무작위로 정하세요.
    2. 다음의 범죄 사실에을 법률 전문가가 아닌 일반인이 물어보듯 자연스럽게 다시 풀어쓰세요.
    3. 이때, 500자 이내로 서술하세요. 그리고 매번 범죄사실이 주어질 때 마다 다른 사람이 질문하듯 감정, 역할, 말투, 법률 지식 수준 등을 바꿔주세요.
    4. 마지막에는 실제로 당신이 담당한 역할을 맡은 사람(피의자, 피해자, 혹은 그 주변인물)이 변호사에게 해당 범죄의 경우 어떤 법령에 의해서 처벌을 받는지 법률상담을 요청하는 다양한 질문들을 덧붙여주세요. 
    5. 1인칭으로 대답하세요.

    범죄 사실은 다음과 같습니다:
    {crime_facts}"""

    query_maker = GPTQueryMaker(prompt=prompt)
    test_data['자연어 쿼리'] = test_data['범죄 사실'].apply(query_maker.generate_simple_query)

    ## 쿼리가 추가된 테스트 데이터를 추가로 저장합니다.
    test_data.to_csv('./datasets/test_data_with_query.csv')
