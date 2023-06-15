import os
import logging
import sys
import pinecone
from fastapi import FastAPI, status
from llama_index import (
    LLMPredictor,
    LangchainEmbedding,
    ServiceContext,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    PromptHelper,
    StorageContext,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import PineconeVectorStore
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
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


def create_service_context():

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_size_limit=chunk_size_limit, chunk_overlap_ratio=0.01)
    llm_predictor = LLMPredictor(llm=AzureOpenAI(temperature=0.5, deployment_name="ohopenai", model_name="gpt-35-turbo", max_tokens=num_outputs))
    embedding_llm = LangchainEmbedding(OpenAIEmbeddings(deployment='text-embedding-ada-002'),embed_batch_size=1)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embedding_llm)
    
    return service_context


def data_ingestion_indexing(directory_path):

    pinecone_index_name = os.environ['PINECONE_INDEX']
    if pinecone_index_name in pinecone.list_indexes():
        pinecone_index = pinecone.Index(index_name=pinecone_index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        store_index = GPTVectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=create_service_context())
    if pinecone_index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=pinecone_index_name,
            dimension=1536,
            metric="cosine"
        )
        pinecone_index = pinecone.Index(index_name=pinecone_index_name)
        documents = SimpleDirectoryReader(directory_path).load_data()
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        store_index = GPTVectorStoreIndex.from_documents(nodes, storage_context=storage_context,service_context=create_service_context())
        
    return store_index

def data_querying(input_text):

    response = index.as_query_engine().query(input_text)
    
    return response.response

iface = gr.Interface(fn=data_querying,
                     server_name="0.0.0.0",
                     inputs=gr.components.Textbox(lines=7, label="输入您的问题"),
                     outputs="text",
                     title="OH Biohealth 健康专家")


index = data_ingestion_indexing("docs")
app = gr.mount_gradio_app(app, iface, path="/expert")

