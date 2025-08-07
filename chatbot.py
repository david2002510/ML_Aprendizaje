#Cargamos las librerias necesarias
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3.2") # Para busquedas off-line

template = """
You're a chatbot that interact with human beings in english or in spanish depending of the prompt

Your answers should be short as possible due you are talking like a human being with your own personality

Here are some conversation history: {context}

Here is the question to answer: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


def chatbot():
    context=""
    print("Welcome to the AI Chatbot! Type 'exit' to quit program.\n")
    print("\n -----------")
    while True: 
        user = input("You:")
        print("\n")
        if user.lower() == "exit":
            break
        result = chain.invoke({"context": context, "question": user})
        print("AI: " + result + "\n")
        context += f"\nUser:{user} AI:{result}"  #Memoria que tiene la IA sobre la conversación

if __name__ == "__main__":
    chatbot()
