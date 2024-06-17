import os
import re
from typing import *

import pandas as pd
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi



class LongformerTokenizer:
    def __init__(self, model_name: str = 'severinsimmler/xlm-roberta-longformer-base-16384') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens



class BM25Retriever:
    def __init__(self, retrieve_df: pd.DataFrame, tokenizer: LongformerTokenizer) -> None:
        self.tokenizer = tokenizer
        self.retrieve_df = retrieve_df
        self.tokenized_facts = [self.tokenizer.tokenize(fact) for fact in retrieve_df['범죄 사실'].tolist()]
        self.bm25 = BM25Okapi(self.tokenized_facts)
        self.search_history: List[Dict[str, any]] = []


    def retrieve(self, query: str, top_n: int = 5) -> Tuple[List[int], List[float]]:
        tokenized_query = self.tokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sorted_indices = bm25_scores.argsort()[::-1][:top_n]
        
        # Store the search results
        self._store_search_results(query, sorted_indices, bm25_scores)

        return sorted_indices.tolist(), bm25_scores.tolist()


    def _store_search_results(self, query: str, indices: List[int], scores: List[float]) -> None:
        result = {
            "query": query,
            "results": [{"index": idx, 
                         "fact": self.retrieve_df['범죄 사실'][idx],
                         'laws': self.retrieve_df['법령의 적용'][idx],
                         "score": scores[idx]} 
                        for idx in indices]
        }
        self.search_history.append(result)


    def get_search_history(self) -> List[Dict[str, any]]:
        return self.search_history
    
    def save_search_history_to_csv(self, file_path: str = './search_history.csv', normalize=True) -> None:
        records = []
        for query_idx, entry in enumerate(self.search_history):
            if normalize:
                query = entry["query"]
                for result in entry["results"]:
                    record = {
                        "query_index": query_idx,
                        "query": query,
                        "document_index": result["index"],
                        "fact": result["fact"],
                        "laws": result["laws"],
                        "score": result["score"]
                    }
                    records.append(record)
            else:
                record = {
                    "query_index": query_idx,
                    "query": entry["query"],
                    "results": entry["results"],
                }
                records.append(record)
                
        df = pd.DataFrame(records)
        df.to_csv(file_path, index=False, encoding='utf-8')
        return None



if __name__ == '__main__':
    
    retrieve_df = pd.read_csv("./datasets/train_data.csv")
    test_df = pd.read_csv("./datasets/test_data_with_query.csv")
    # DocumentLoader 인스턴스 생성 및 문서 로드

    # LongformerTokenizer 인스턴스 생성
    tokenizer = LongformerTokenizer(model_name='severinsimmler/xlm-roberta-longformer-base-16384')
    tokenized_documnets = [tokenizer.tokenize(document) for document in retrieve_df]

    # BM25Retriever 인스턴스 생성
    retriever = BM25Retriever(retrieve_df, tokenizer)

    # 쿼리 예제: doc5를 참고해서 음주운전 + 과실치사.
    test_sample = test_df.iloc[4, :]
    query_gpt = test_sample['자연어 쿼리']
    query_handwritten= """어젯밤에 제가 퇴근하고 동네에서 친구랑 같이 술을 한 잔 했는데요. 시간이 늦어서인지 아무리 기다려도 대리가 안오더라구요.
    그래서 어쩔 수 없이 제가 운전대를 잡고, 후면주차시켰던 차를 뺴려는데, 하필 그때 차 뒤에 저랑 같이 술을 먹던 친구가 있던거에요...
    제가 차를 조금 급하게 빼던 중이라서 친구가 차에 좀 세게 치였고, 지금은 병원에서 입원 중입니다.
    이런 경우에도 제가 처벌받을 수 있나요? 친구인데 어떻게 좀 안될까요?"""


    # 문서 검색
    sorted_indices, bm25_scores = retriever.retrieve(query_handwritten)
    sorted_indices, bm25_scores = retriever.retrieve(query_gpt)

    retriever.save_search_history_to_csv()
    print('the result is saved as serach_result.csv file')