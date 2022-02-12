import streamlit as st
import pandas as pd
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xlrd
st.write("""
# 과제간 유사도 분석 : 노이즈 제거용 
""")
expander_bar1 = st.expander("About", expanded=True)
expander_bar1.markdown("""
* 파일 필드에 **<과제명>**과 **<연구내용>** 필드가 반드시 있어야 합니다. 
* 없으면 파일에 동일하게 필드명을 바꿔주세요.(확인 !!)
""")

### input features in the sidebar
st.sidebar.header('User Input Features')

input_df = pd.read_excel('키워드분석용_rawdata.xlsx')
rank = 100
title = st.sidebar.text_area('대표 과제명 입력', "도시 공간 유형별 미세먼지 저감 모델 개발 및 실증 연구")


# 2. 연구내용으로 추천시스템 만들기
stop_words = []
with open('한국어불용어100.txt', 'r', encoding='utf-8') as f:
    stop_words.append(f.readlines())
stop_words_new = [words.split('\t')[0] for words in stop_words[0]]

stop_words_new