#Cargamos las librerias necesarias
from langchain_community.document_loaders import WikipediaLoader

while True:
    print("\n\n -----------")
    question = input("¿Que consulta de Wikipedia desea realizar? (q to quit)\n")
    print("\n")
    if question == "q":
        break
    docs = WikipediaLoader(question, load_max_docs=2).load()
    metadata = docs[0].metadata  # metadata of the first document
    result = docs[0].page_content[:400]  # a part of the page content
    print(result)
