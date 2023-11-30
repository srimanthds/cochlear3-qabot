#!/usr/bin/env python
# coding: utf-8

# In[15]:


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from pymongo import MongoClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
import os
import shutil
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pymongo
import joblib
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import pandas as pd



# In[2]:


PDF_FOLDER_PATH = "Data/"
LOADED_PDF_FILES_PICKLE = "loaded_pdf_files_pickle.pkl"
VECTOR_SEARCH_PICKLE = "vector_search_pickle.pkl"
DB_NAME = "cochlear_7"
COLLECTION_NAME = "vectorSearch"
INDEX_NAME = "default"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# In[3]:


def get_secret_key():
    open_api_key = st.secrets.open_api_key
    if not open_api_key:
        raise ValueError("The open_api_key environment variable is not set.")
    s1 = st.secrets.db_username
    s2 = st.secrets.db_pswd
    atlas_connection_string = "mongodb+srv://{s1}:{s2}@cluster0.1thtla4.mongodb.net/?retryWrites=true&w=majority".format(s1 = s1, s2 = s2)
    if not atlas_connection_string:
        raise ValueError("The atlas_connection_string environment variable is not set.")
    secret_key_dict = {"open_api_key": open_api_key, "atlas_connection_string": atlas_connection_string}
    return secret_key_dict


# In[4]:


def get_vector_search_object(cluster,db_name,collection_name, index_name,open_api_key):
    mongodb_collection = cluster[db_name][collection_name]
    doc =  Document(page_content="dummy text", metadata={"source": "dummy"})
    vector_search = MongoDBAtlasVectorSearch.from_documents(
                    documents=[doc],
                    embedding=OpenAIEmbeddings(api_key=open_api_key),
                    collection=mongodb_collection,
                    index_name=index_name 
                )
    return vector_search


# In[5]:


def connect_mongodb(atlas_connection_string):
    cluster = MongoClient(atlas_connection_string)
    try:
        cluster.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return cluster


# In[17]:


def get_prompt():
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Be precise and accurate and be logical in answering. 

    Your job is to first reply to the question giving the reason too

    While formulating it be accurate and logical. Do not give contradicting answers. 

    The context that you use to answer the question should be the only facts you will look out for and not any other external
    facts. While formulating the response read the question again and answer accordingly to avoid contradicting replies

    {context}

    Question: {question}
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt

def get_prompt_critique():
    prompt_template = """You are the smart engine that looks at the response below along with the question asked
    and makes edit to the response only if you think the response needs to be edited due to logical or contradicting mistakes

    1. First read the question stated below and understand it.
    2. Read the response below. This response acts as the answer to the question. However this response may be semantically
    or logically incorrect in response.
    3. The response usually will have 2 parts, the first part will be the answer and the second part will have the context 
    or information or reasoning from which the answer was stated.
    4. If the answer and the reason are not in alignment, reformulate the response and send the correct response again

    
    Be precise and accurate and be logical in answering. 
    
    While formulating it be accurate and logical. Do not give contradicting answers. 

    The response should be the only facts you will look out for and not any other external
    facts. While formulating the response read the question again and answer accordingly to avoid contradicting replies

    Reply with the reformulated response.

    Just send the response, do not prefix with anything like "Response :" or "Revised Response :"

    Question: {Question}
    
    Response: {Response}


    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["Question", "Response"]
    )
    return prompt

# In[20]:


def get_response(db_name, collection_name, index_name, query):
    secret_key_dict = get_secret_key()
    open_api_key = secret_key_dict["open_api_key"]
    atlas_connection_string = secret_key_dict["atlas_connection_string"]
    cluster = connect_mongodb(atlas_connection_string)
    vector_search = get_vector_search_object(cluster,db_name,collection_name, index_name, open_api_key)
    qa_retriever = vector_search.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "post_filter_pipeline": [{"$limit": 25}]},
    )
    prompt = get_prompt()
    try:
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(api_key=open_api_key,temperature=0),
            chain_type="stuff",
            retriever=qa_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
    except:
        time.sleep(120)
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(api_key=open_api_key,temperature=0),
            chain_type="stuff",
            retriever=qa_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    docs = qa({"query": query})

    # print(docs["result"])
    # print(docs["source_documents"])
    return docs


# In[ ]:

result = []
# Page title
st.set_page_config(page_title='Cochlear Smart QA Engine')
st.title('Cochlear Smart QA Engine')

# # File upload
# uploaded_file = st.file_uploader('Upload an article', type='pdf')
# print(dir(uploaded_file))
# Query text

secret_key_dict = get_secret_key()
open_api_key = secret_key_dict["open_api_key"]

if 'qa_data' not in st.session_state:
    st.session_state.qa_data = {'question': '', 'responses': []}

streamlit_pwd = st.secrets.streamlit_pwd
# Form input and query


user_input = st.text_input('Enter the application password:', type='password')
if user_input != streamlit_pwd:
    st.error("Authentication failed. Please provide a valid API key.")
else:
    with st.form('myform', clear_on_submit=True):
        query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=False)

        # openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
        submitted = st.form_submit_button('Submit')
        
        if submitted:
            with st.spinner('Calculating...'):
                try:
                    docs = get_response(DB_NAME,COLLECTION_NAME,INDEX_NAME,query_text)
                except:
                    time.sleep(120)
                    docs = get_response(DB_NAME,COLLECTION_NAME,INDEX_NAME,query_text)

                response = docs["result"]
                try:
                    prompt = get_prompt_critique()
                    llm = OpenAI(api_key=open_api_key,temperature=0)
                    prompt.format(Question=query_text,Response=response)
                    chain1 = LLMChain(llm=llm,prompt=prompt)
                    response = chain1.run(Question=query_text,Response=response)
                except:
                    time.sleep(120)
                    prompt = get_prompt_critique()
                    llm = OpenAI(api_key=open_api_key,temperature=0)
                    prompt.format(Question=query_text,Response=response)
                    chain1 = LLMChain(llm=llm,prompt=prompt)
                    response = chain1.run(Question=query_text,Response=response)
                    
                result.append(response)
                st.session_state.qa_data['question'] = query_text
                st.session_state.qa_data['responses'].append(response)
                for idx, r in enumerate(st.session_state.qa_data['responses'][::-1], start=1):
                    st.info(f"Response : {r}")
                st.title('Top Similar Documents')
                df_lis = []
                for i in docs["source_documents"]:
                    lis = []
                    lis.append(i.page_content)
                    lis.append(i.metadata["source"])
                    lis.append(i.metadata["page"])
                    df_lis.append(lis)
                similar_df = pd.DataFrame(df_lis,columns = ["Text", "Source Document", "Page Number"])

                st.table(similar_df)
                
                
    #             del openai_api_key
    st.write(f"Last Submitted Question: {st.session_state.qa_data['question']}")
    st.write("All Responses:")
    for idx, r in enumerate(st.session_state.qa_data['responses'], start=1):
        st.write(f"Response {idx}: {r}")
        # if len(result):
        #     st.info(response)

