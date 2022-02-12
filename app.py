import streamlit as st
import pandas as pd
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# 0. 파일 열기
uploaded_file = st.sidebar.file_uploader("Upload your excel file", type=["xlsx"])
try: 
    input_df = pd.read_excel(uploaded_file)
    for name in input_df.columns:
        if "Unnamed" in name:
            del input_df[name]
except:
    input_df = None

# 1. 유사 프로젝트 수와 대표과제명 지정
st.sidebar.write("""***""")
st.sidebar.subheader('유사과제 갯수 설정')
rank = st.sidebar.text_input('과제수', 100)
st.sidebar.write("""***""")
st.sidebar.write("""### 과제명 입력
* 과제간 유사도를 구하기 위해 기준이 될 과제명을 설정해야 됩니다.
""")
title = st.sidebar.text_area('대표 과제명 입력', "도시 공간 유형별 미세먼지 저감 모델 개발 및 실증 연구")


# 2. 연구내용으로 추천시스템 만들기
stop_words = []
with open('한국어불용어100.txt', 'r', encoding='utf-8') as f:
    stop_words.append(f.readlines())
stop_words_new = [words.split('\t')[0] for words in stop_words[0]]

if input_df is not None:
    input_df['연구내용'] = input_df['연구내용'].fillna('')
    input_df['연구내용'] = input_df['연구내용'].astype('str')
    tfidf = TfidfVectorizer(stop_words=stop_words_new)
    tfidf_matrix_content = tfidf.fit_transform(input_df['연구내용'])
    st.write("입력데이터 과제수는 ", tfidf_matrix_content.shape[0], "개입니다.")

    ### 2-1. 유사도 행렬
    cosine_sim_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    ### 과제명을 key, 과제 인텍스를 value로 하는 딕셔너리 만들기
    title_to_index = dict(zip(input_df['과제명'], input_df.index))

    ## 2-2. 추천시스템 함수
    @st.cache
    def get_recommdataions1(title, rank, cosine_sim=cosine_sim_content):
        idx = title_to_index[title]
        sim_scores = list(enumerate(cosine_sim[idx]))  # 유사도 가져오기
        input_df['sim_score'] = [sim for index, sim in sim_scores]

        # 상위 랭크 추출
        sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)  # 유사도 정렬
        sim_scores = sim_scores[1:rank + 1]
        project_indices = [idx[0] for idx in sim_scores]
        
        return input_df, input_df['과제명'].iloc[project_indices]

    ## 2-3. 파일 다운로드 함수
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    def filedownload(df):
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv.encode(encoding='utf-8-sig')).decode(encoding='utf-8-sig')
        href = f'<a href="data:file/csv;base64,{b64}" download="similar_projects.csv">Download CSV File</a>'
        return href

    ## 2-4. 결과 출력
    df, titles = get_recommdataions1(title, int(rank))
    st.markdown("""#### 1. 유사과제 상위 10개 과제명
    나머지는 파일을 다운로드해서 확인해 보세요!!""")
    st.write(titles.values[:10])

    # Download dataframe
    st.write("""---""")
    st.write("""#### 2. 파일 다운로드""")
    st.write("""유사도 정보를 추가한 파일 다운로드(필드명 : sim_score). 
    유사도는 1에서 -1까지 범위가 나올 수 있으며, 
    1에 가까울 수록 유사도가 높고 0 이하이면 성격이 다른 과제라고 기본적으로 해석합니다.""")
    st.markdown(filedownload(df), unsafe_allow_html=True)
    st.write("""***""")