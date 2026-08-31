from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an anime recommendation assistant.

Use the following anime information to answer the user's question.

Relevant anime:
{anime}

Question:
{question}

Answer based only on the information provided above.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n-------------------------------------")
    question = input("Ask your Question (q to quit): ")
    print("\n\n-------------------------------------")

    if question.lower() == "q":
        break

    anime = retriever.invoke(question)

    result = chain.invoke({
        "anime": anime,
        "question": question
    })

    print(result)