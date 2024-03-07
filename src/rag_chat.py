#TODO
# document sources for chunks
# web interface (see video)
# compare performance vs vanilla network, perhaps test rag on unseen scientific papers

#pip install --quiet langchain_community langchain gpt4all chromadb unstructured tiktoken gradio

#import pprint

import os

import gradio as gr
from gradio.themes.base import Base

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def create_vector_store(db_directory, data_directory, embedding):
    print("---CREATING VECTOR STORE (this may take a while)---")
    loader = DirectoryLoader(data_directory)  # , glob="**/*.txt")
    docs = loader.load()

    # split text into chunks with overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    # create vector store and index
    vectorstore = Chroma.from_documents(documents=splits, collection_name="rag-chroma", embedding=embedding, persist_directory=db_directory)

    return vectorstore.as_retriever()


def fetch_vector_store(db_directory, embedding):
    print("---FETCHING VECTOR STORE---")
    vectorstore = Chroma(collection_name="rag-chroma", embedding_function=embedding, persist_directory=db_directory) # Chroma(persist_directory=db_directory, embedding_function=embedding)
    return vectorstore.as_retriever()


def retrieve(retrieving, question):
    print("---RETRIEVE---")
    documents = retrieving.get_relevant_documents(question)
    return documents


def context_formatting(documents):
    content = ""
    for index, document in enumerate(documents):
        content = content + "[doc" + str(index+1) + "]=" + document.page_content.replace("\n", " ") + "\n\n"
    return content


def source_formatting(documents):
    sources = ""
    for index, document in enumerate(documents):
        sources = sources + "[doc" + str(index+1) + "]=" + document.metadata["source"] + "\n\n"
    return sources


def generate(question, documents, use_llm):
    print("---GENERATE---")
    # adapted from https://smith.langchain.com/hub/rlm/rag-prompt
    rag_prompt = ChatPromptTemplate.from_template("You are an assistant for question-answering tasks. Use the "
                                                  "following pieces of retrieved context to answer the question. If "
                                                  "you don't know the answer, just say that you don't know. Keep the "
                                                  "answer concise, truthful, and informative. If you decide to use a "
                                                  "source, you must mention in which document you found specific "
                                                  "information. Sources are indicated in the context by "
                                                  "[doc<doc_number>].\n"
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

    #input_question2 = "How do you compile regular expressions in Python?"
    #input_question = "Explain the None type in Python."
    #docs = retrieve(retriever, input_question)
    #output = generate(input_question, context_formatting(docs), use_llm)

    def complete_rag(question):
        docs = retrieve(retriever, question)
        output = generate(question, context_formatting(docs), use_llm)
        return source_formatting(docs), output

    with gr.Blocks(theme=Base(), title="Q&A on your data with RAG") as demo:
        gr.Markdown("# Q&A on your data with RAG")
        textbox = gr.Textbox(label="Question:")
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
        with gr.Column():
            output1 = gr.Textbox(lines=1, max_lines=10, label="Sources")
            output2 = gr.Textbox(lines=1, max_lines=10, label="Generated output by LLM, incorporating retrieved documents")

        button.click(complete_rag, textbox, outputs=[output1, output2])

    demo.launch()

    #pprint.pprint(output)

    #print(output)
    #print(source_formatting(docs))

    #print(docs)
    #print(source_formatting(docs))
    #print(context_formatting(docs))