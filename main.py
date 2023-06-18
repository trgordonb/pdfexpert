import os
import logging
import sys
import pinecone
from fastapi import FastAPI, status
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
import gradio as gr

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENV'])
app = FastAPI()
index = None

@app.get('/')
def hello_world():
    return 'hello world'

@app.get('/healthz', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    return {'healthcheck': 'Everything OK!'}

def data_ingestion_indexing(directory_path):
    pinecone_index_name = os.environ['PINECONE_INDEX']
    embeddings = OpenAIEmbeddings(deployment='text-embedding-ada-002', chunk_size=1)

    store_index = None
    if pinecone_index_name in pinecone.list_indexes():
        store_index = Pinecone.from_existing_index(pinecone_index_name, embeddings)
    elif pinecone_index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=pinecone_index_name,
            dimension=1536,
            metric="cosine"
        )
        loader = PyPDFLoader(f"{directory_path}/OHTechExplained.pdf")
        pages = loader.load_and_split()
        embeddings = OpenAIEmbeddings(deployment='text-embedding-ada-002', chunk_size=1)
        store_index = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index_name)
        
    return store_index

def data_querying(input_text):
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_version=os.getenv('OPENAI_API_VERSION'),
        deployment_name='ohopenai',
        temperature=0.0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever()
    )

    result = qa.run(input_text)
    return result

iface = gr.Interface(fn=data_querying,
                     server_name="0.0.0.0",
                     inputs=gr.components.Textbox(lines=7, label="输入您的问题"),
                     outputs="text",
                     title="OH Biohealth 健康专家")


index = data_ingestion_indexing("docs")
app = gr.mount_gradio_app(app, iface, path="/expert")

