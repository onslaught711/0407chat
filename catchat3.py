#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# streamlit 가동 방법
# 1. 해당 파일을 .py로 다운로드 한다.
# 2. 커맨드창을 열어서 py파일이 있는 곳까지 이동한다.
# 3. streamlit run OO.py 실행

# In[1]:


#!pip install streamlit


# In[3]:


#!pip install streamlit_chat


# In[4]:


import streamlit as st

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import (StuffDocumentsChain, LLMChain,
                              ConversationalRetrievalChain)
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler

from openai import OpenAI
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from langchain.schema import LLMResult
import requests
import tempfile
from PIL import Image
from io import BytesIO
    

api_key = st.secrets["default"]["api_key"] #앱을 올릴때는 이걸로 해야함(안그럼 사라짐)





class StreamHandler(BaseCallbackHandler): #지피티마냥 대답을 실시간으로 할 수 있게 (건드리지 말것) #그게 아니면 완성본 보여주는 handler
    def __init__(self, container, initial_text="", display_method='markdown'):
        super().__init__()
        self.container = container
        self.text = initial_text
        self.display_method = display_method
        self.complete_response = None  # 완성된 답변을 저장하는 변수 추가

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # LLM 처리가 끝났을 때 호출됩니다.
        
        # LLMResult 인스턴스에서 'generations' 키의 값에 접근
        generations = response.dict().get('generations', [])

        # 'generations' 내부의 첫 번째 요소에서 첫 번째 딕셔너리를 가져옴
        if generations and len(generations) > 0 and len(generations[0]) > 0:
            first_generation = generations[0][0]  # 첫 번째 'generation'의 첫 번째 항목

            # 'message' 키의 'content' 값에 접근
            content = first_generation.get('message', {}).get('content', '')
        
        self.complete_response = content  # 최종 텍스트를 저장합니다.
        

#웹에서 이미지를 가져와야함        
image_path = "https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMzExMjlfMTky%2FMDAxNzAxMjMxNTA3NTQ0.TMBl7yxz5R7xvDrjyMZs5_CN04135Y08YEGYl3YWKY0g.Ha91B4S7m8m31B-S6kzP_9i67lPfuRXdB1A9WK4mKU8g.JPEG.a3574284%2F3472511939180767549_20231123234200265.jpg&type=sc960_832"
# 이미지 데이터를 가져옴
response = requests.get(image_path) 

# BytesIO 객체를 사용하여 바이너리 스트림을 생성
image_stream = BytesIO(response.content)

# PIL로 이미지 열기
image = Image.open(image_stream)

#이 위까진 안바꿔도 됨

img = image.resize((300, 300)) #사이즈 조정은 원하는대로

# Streamlit 앱에 사진 추가
st.image(img, caption='애옹~')


# Set up the title and Streamlit session state for managing chat history and model selection
st.title("집사가 되어줘")

if "openai_model" not in st.session_state: #원하는 버전 작성
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state['messages'] = [] #신경 안써도 됨
    
# Google Drive에서 PDF 파일의 다운로드 링크 생성
file_id = '1AZwhyGn0U-_z85qpQe1qQtgvpsUb2CoG'
pdf_url = f'https://drive.google.com/uc?id={file_id}&export=download'

#https://drive.google.com/file/d/1AZwhyGn0U-_z85qpQe1qQtgvpsUb2CoG/view?usp=sharing
#가운데 키처럼 생긴것만 file_id에 넣어주면 됨

response = requests.get(pdf_url)

if response.status_code == 200: #200은 성공이라는 것! (안전장치로 response가 잘 됑스면.. 인거)
    # BytesIO 객체를 사용하여 PDF 파일로부터 데이터를 로드
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    # Extract data from PDF
    loader = PyPDFLoader(tmp_file_path) #웹페이지 하고싶으면 이걸 로더로
    data = loader.load()

    # Generate document vectors
    embeddings = OpenAIEmbeddings(api_key=api_key) #pdf와 같은 로컬을 참조하게 하려면 구글 자주 사용(전체허용해서)
    vectors = FAISS.from_documents(data, embeddings)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
#여기까진 바꿀게 없다

#이 아래부터 멘트인걸 바꾸기
    
    st.chat_message("user").write('야옹~') #user-> 내가 보내는 메시지 써놓기 (지워도댐)
    st.chat_message("assistant").write('애옹~ 도시혁신 프로젝트에 대해 질문이 있어?')
    
    # Display chat history
    for message_type, message_text in st.session_state['messages']:
        if message_type == "user":
            st.chat_message("user").write(message_text)
        else:  # 'assistant'
            st.chat_message("assistant").write(message_text)


    # Process new chat inputs
    if prompt := st.chat_input("질문을 적어주세요."): #흐릿하게 적혀져있는..멘트
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):  #assistant가 대답을 할 차례에 아래처럼 대답을 해라 지정
        
            stream_handler = StreamHandler(st.empty(), display_method='markdown')

            # Create conversational retrieval chain  #llm 모델 지정
            qa = RetrievalQA.from_llm(llm=ChatOpenAI(
                streaming=True,
                callbacks=[stream_handler], #stream 형태로 실시간으로 보내주라
                temperature=0.0, #조정 가능 (fact기반 - 창의적 대답)
                model_name='gpt-4-0125-preview', #쓸 모델 네임 적어주기
                openai_api_key=api_key),
                retriever=vectors.as_retriever()) #검색을 가능하게

            qa(prompt)
            
            end = True
            
            # StreamHandler에서 완성된 답변을 가져와 st.session_state에 저장
            if end:
                # 사용자 메시지 저장
                st.session_state['messages'].append(("user", prompt))
                # 어시스턴트 메시지 저장
                st.session_state['messages'].append(("assistant", stream_handler.complete_response))
   


# In[5]:


file_id = '1AZwhyGn0U-_z85qpQe1qQtgvpsUb2CoG'
pdf_url = f'https://drive.google.com/uc?id={file_id}&export=download'

print(pdf_url)


# In[5]:


#file_id = '1AZwhyGn0U-_z85qpQe1qQtgvpsUb2CoG'
#pdf_url = f'https://drive.google.com/uc?id={file_id}&export=download'

#print(pdf_url)

#아래 링크 들어가서 바로 다운되면 된거


# In[ ]:




