from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pywin.framework.sgrepmdi import GrepView

from vector import retriever # This is the retriever we created in vector.py and used to retrieve the most relevant reviews

model = OllamaLLM(model = "llama3.1")

template = """
you are an expert answering questions about pizza restaurant
Here are some relevant reviews: {reviews}
Here is the question to be answered: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
# invoke this train to run multiple things together
chain = prompt | model


while True:
    print("\n\n -----------------------------------------")
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    # retrieve the most relevant reviews
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)