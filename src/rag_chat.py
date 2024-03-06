#!pip install --quiet langchain_community tiktoken langchain-openai langchainhub chromadb langchain langgraph tavily-python langchain-mistralai gpt4all

import pprint

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate


def create_vector_store(db_directory, data_directory, embedding):
    print("---CREATING VECTOR STORE (this may take a while)---")
    loader = DirectoryLoader(data_directory)  # , glob="**/*.txt")
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(docs)

    # Index
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        collection_name="rag-chroma",
        embedding=embedding,
        persist_directory=db_directory
    )
    return vectorstore.as_retriever()


def fetch_vector_store(db_directory, embedding):
    print("---FETCHING VECTOR STORE---")
    vectorstore = Chroma(persist_directory=db_directory, embedding_function=embedding)
    return vectorstore.as_retriever()


def retrieve(question):
    print("---RETRIEVE---")
    documents = retriever.get_relevant_documents(question)
    return documents


def generate(question, documents, use_llm):
    print("---GENERATE---")
    # adapted from https://smith.langchain.com/hub/rlm/rag-prompt
    rag_prompt = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the "
                                                  "following pieces of retrieved context to answer the question. If "
                                                  "you don't know the answer, just say that you don't know. Keep the "
                                                  "answer concise, truthful, and informative.\n"
                                                  "Question: {question} \n"
                                                  "Context: {context} \n"
                                                  "Answer:")

    llm = ChatOllama(model=use_llm, temperature=0)
    chain = rag_prompt | llm | StrOutputParser()
    output = chain.invoke({"context": documents, "question": question})
    return output


if __name__ == "__main__":
    print("---STARTING PROGRAM---")
    use_llm = "mistral:instruct"
    embedding = GPT4AllEmbeddings()
    data_directory = '../data'
    db_directory = './chroma_db'
    retriever = None
    # creating or fetching vector store
    if not os.path.isdir(db_directory):
        retriever = create_vector_store(db_directory, data_directory, embedding)
    else:
        retriever = fetch_vector_store(db_directory, embedding)

    input_question = "How do you compile regular expressions in Python?"
    docs = retrieve(input_question)
    output = generate(input_question, docs, use_llm)

    #pprint.pprint(output)

    print(output)