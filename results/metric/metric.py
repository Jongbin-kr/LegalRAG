# 필요한 패키지 import
import pandas as pd
import re
from datetime import datetime

# 파일 경로 지정
# file_path = './baseline3_AlphaMist_result.csv'
file_path = './baseline3_gpt_result.csv'

# CSV 파일 로드
df = pd.read_csv(file_path,encoding='utf-8')

# 법령명 추출 함수
def extract_law_names(text):
    # 입력값이 NaN이나 None인 경우 빈 문자열로 처리
    if pd.isna(text):
        return []
    # 정규 표현식으로 법령명 추출
    law_names = re.findall(r'\b(?:[\w가-힣]+법|법률|규칙|조례|명령|규정)\s제[\d가-힣]+조\b', str(text))
    return law_names

# 열 이름 확인 및 선택
if '응답' in df.columns:
    column_data = df['응답']
elif '답변' in df.columns:
    column_data = df['답변']
else:
    print("해당하는 열이 없습니다.")
    column_data = None
# '답변' 컬럼에서 법령명 추출
df['extracted_laws'] = column_data.apply(extract_law_names)

# 현재 날짜와 시간 가져오기
now = datetime.now()
formatted_date = now.strftime('%Y%m%d%H%M%S')

# 원본 파일명에서 확장자를 제외한 부분 가져오기
base_filename = file_path.split('/')[-1].split('.')[0]



# 결과 CSV 파일로 저장
output_file_path = f'{formatted_date}:{base_filename}.csv'  # 저장할 파일 이름
df.to_csv(output_file_path, index=False)

# 저장된 파일 경로 출력
print(f'Data saved to {output_file_path}')
