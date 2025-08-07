from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd 
import csv


df = pd.read_csv("Encuesta_CSV.csv")
print(df)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"

add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i,j in df.iterrows():
        document = Document(
            page_content=j["Title"] + " " + j["Review"] ,
            metadata={"name": j["Name"], "date": j["Date"]}, 
            id=str(i)
        )
        ids.append(id)
        documents.append(document)

vector_store = Chroma(
    collection_name="review-productos",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents,ids=ids)

retriever = vector_store.as_retriever( # RAG
    search_kwargs={"k":5}
)