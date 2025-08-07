#Cargamos las librerias necesarias
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever #RAG from vector.py

model = OllamaLLM(model="llama3.2")

template = """
You are an exeprt in answering questions about the Reviews of the products 

You only know about your database, no provide any information of third-party corporations

You can't say you dont know what review is, you should know

You shouldn't mentioning the name of reviewers , they are anonymus.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n -----------")
    question = input("¿Que consulta sobre nuestra base de datos desea realizar? (q to quit)\n")
    print("\n\n")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
