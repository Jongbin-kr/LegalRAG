import pandas as pd
import re


def extract_law_names(input_text):
    # 법명, 조항, 항, 단서, 호를 포함하여 추출하는 정규 표현식
    pattern = r'([가-힣]+법)\s*제(\d+조)\s*(제\d+항)?\s*(단서)?\s*(제\d+호)?'
    
    # 정규 표현식을 사용하여 법명 및 상세 조항 추출
    matches = re.findall(pattern, input_text)
    
    # 각 매치는 튜플 형태로 반환되며, 튜플 내 빈 문자열을 제거하고 합쳐서 표준 형태로 정리
    processed_matches = [' '.join(filter(None, match)) for match in matches]

    # 중복 제거를 위해 set 사용
    unique_laws = set(processed_matches)

    # 정렬된 결과 반환
    return ', '.join(sorted(unique_laws))


# 파일 읽기
data = pd.read_csv('/Users/eumhyeyoung/Desktop/collection/development/Github/Hyeyoung-Eum2/1.project/COSE461NLP/preprocessing2/240522_total.csv', encoding='utf-8')

# 두 번째 열의 데이터 추출
input_texts = data.iloc[:, 2]

# 법명 추출
output_laws = input_texts.apply(extract_law_names)

# 결과 데이터프레임 생성
result_df = pd.DataFrame({
    'Original Text': input_texts,
    'Extracted Laws': output_laws
})

# 결과를 CSV 파일로 저장
result_df.to_csv('./240524_extracted_laws.csv', index=False, encoding='utf-8')

print("CSV 파일이 성공적으로 저장되었습니다.")
