## CASENOTE.kr Crawler

[CASENOTE.kr](https://casenote.kr/)에서 '교통사고 고단'이라고 검색한 판례들로부터 범죄 사실과 법령의 적용 부분을 크롤링한 뒤, 해당 판례에 적용된 법령 정보들을 GPT-4o로 생성해 추가하였습니다.

구체적인 크롤링 과정은 다음과 같습니다.
1. bs4와 requests를 활용해서 데이터셋을 크롤링했습니다.(`crawling.py`)
2. 간단한 전처리를 거쳤습니다. (`preprocessing.ipynb`)
3. 해당 판례의 법령의 적용 부분에서, 법령명에 관련된 부분들만 추출했습니다. (`extract_law_names.py`)
4. 이후, GPT-4o를 활용해 법령명을 정제하고, 해당 법령에 대한 내용들을 사전 형태로 추가했습니다.