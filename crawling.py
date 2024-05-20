import requests
from bs4 import BeautifulSoup
import re
import csv
import time
from datetime import datetime

# 소제목을 인식하기 위한 패턴 정의
title_patterns = [
    '범죄사실', '증거의 요지', '법령의 적용', '양형의 이유', '무죄 부분',
    '결론', '판단', '이 사건 공소사실'
]

# 소제목을 위한 정규 표현식 컴파일
def compile_patterns():
    title_regex = re.compile(r'\b(?:{})\b'.format('|'.join(title_patterns)), re.IGNORECASE)
    crime_fact_regex = re.compile(r'\b범\s*죄\s*사\s*실\b', re.IGNORECASE)
    return title_regex, crime_fact_regex

title_regex, crime_fact_regex = compile_patterns()

# CSV 파일로 데이터 저장
def save_data_to_csv(data, skipped_urls, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        if skipped_urls:
            writer.writerow(['Skipped URLs'])
            for url in skipped_urls:
                writer.writerow([url])
    print(f"데이터가 '{filename}'에 저장되었습니다.")

# URL 요청 및 재시도 로직
def fetch_url(url, headers, retries=5, delay=1):
    for attempt in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        elif response.status_code == 429:
            print(f"Rate limit reached, retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        else:
            print(f"Failed to retrieve data, status code: {response.status_code}")
            break
    return None

current_time = datetime.now().strftime("%m월%d일%H시%M분")

start_page = 1
last_page = 20
###이 부분 적절하게 조정###
search_keyword = '교통사고 고단'
start_year = 2010
last_year = 2013
required_word_in_page_title = '고단' #교통사고 고단이라고 입력했는데도, 고단이 아닌 다른 결과가 나오는 경우를 방지하기 위해 넣었음.
###이 부분 적절하게 조정###

hrefs = []
joined_keyword = '교통사고 고단'.replace(' ', '')

filename = f'[{current_time}]{joined_keyword}({start_year}~{last_year}).csv'

headers = {
    'Referer': 'https://casenote.kr/%EB%8C%80%EA%B5%AC%EC%A7%80%EB%B0%A9%EB%B2%95%EC%9B%90/2023%EA%B3%A0%EB%8B%A82597',
    'Sec-Ch-Ua': '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"macOS"',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
}

# URL 수집
for i in range(start_page, last_page + 1):
    url = f'https://casenote.kr/search/?q={search_keyword}&sort=0&period=4&period_from={start_year}&period_to={last_year}&partial=0&page={i}'
    response = fetch_url(url, headers)
    if response:
        soup = BeautifulSoup(response.text, 'html.parser')
        for a_tag in soup.find_all('a', class_='casename'):
            hrefs.append(a_tag.get('href'))

data = [['url', 'PageTitle', 'Title'] + title_patterns]  # CSV 헤더
skipped_urls = []

try:
    href_index = 0
    for href in hrefs:
        href_index += 1
        print(f"Processing {href_index}/{len(hrefs)}:{href}")
        url2 = f'https://casenote.kr/{href}'
        response = fetch_url(url2, headers)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_title = soup.find('title').text.strip()
            if required_word_in_page_title not in page_title:
                skipped_urls.append(url2)
                continue
            title = soup.find('h1').text.strip()
            row = [url2, page_title, title]
            text_content = {}
            current_section = None
            section_texts = []
            for p in soup.find_all('p', class_='main-sentence'):
                text = p.get_text().strip()
                if title_regex.search(text) or crime_fact_regex.search(text):
                    text = '범죄사실' if crime_fact_regex.search(text) else text
                    if current_section:
                        text_content[current_section] = '\n'.join(section_texts)
                    current_section = text
                    section_texts = []
                else:
                    section_texts.append(text)
            if current_section:
                text_content[current_section] = '\n'.join(section_texts)
            for pattern in title_patterns:
                row.append(text_content.get(pattern, ''))
            data.append(row)
except Exception as e:
    print("크롤링 중 오류 발생:", e)
finally:
    save_data_to_csv(data, skipped_urls, filename)
