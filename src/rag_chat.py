__author__ = "Vinzenz Richard Ulrich"

# pip install gradio langchain gpt4all chromadb pypdf tiktoken
# pip install --quiet gradio langchain gpt4all chromadb pypdf tiktoken


# imports
import os

import gradio as gr
from gradio.themes.base import Base

import glob

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def chunks_from_pdf(pdf_directory):
    """
    Chunks all pdfs from a directory
    :param pdf_directory: directory of pdfs
    :return: list of chunks
    """
    # fetching all pdfs from the directory and storing them as strings in a list
    docs = []
    for file in glob.glob(pdf_directory + "/*.pdf"):
        loader = PyPDFLoader(file)
        doc = loader.load()
        docs.extend(doc)

    # split texts into chunks with overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    return splits


def chunks_from_text(text_directory):
    """
    Chunks all text files from a directory
    :param text_directory: directory of text files
    :return: list of chunks
    """
    # fetch all txt files from the firectory and store them in a list
    loader = DirectoryLoader(text_directory, loader_cls=TextLoader)  # , glob="**/*.txt")
    docs = loader.load()

    # split texts into chunks with overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    return splits


def chunking(data_directory):
    """
    Automatically calls the correct chunking function, either for pdfs or for txt files
    :param data_directory: directory of data, either ../pdf or ../text
    :return: result from the corresponding chunking function
    """
    if data_directory == "../pdf":
        return chunks_from_pdf(data_directory)
    else:
        return chunks_from_text(data_directory)


def create_vector_store(db_directory, chunks, embedding):
    """
    Creates a chromaDB vector embedding store for all chunks of the data
    :param db_directory: directory to persistently store the resulting vector store
    :param chunks: list of chunks of data
    :param embedding: embedding function
    :return: retriever on vector store
    """
    print("Creating vector store (this may take a while)")

    # create vector store and index
    vectorstore = Chroma.from_documents(documents=chunks, collection_name="chromemwah", embedding=embedding,
                                        persist_directory=db_directory)

    return vectorstore.as_retriever()


def fetch_vector_store(db_directory, embedding):
    """
    Fetches a chromaDB vector embedding store of the data
    :param db_directory: directory where vector store is persistently stored
    :param embedding: embedding function
    :return: retriever on vector store
    """
    print("Fetching vector store")
    vectorstore = Chroma(collection_name="chromemwah", embedding_function=embedding, persist_directory=db_directory)
    return vectorstore.as_retriever()


def retrieve(retrieving, question):
    """
    Retrieve relevant documents from vector store based on query/question
    :param retrieving: retriever
    :param question: user query
    :return: relevant documents
    """
    print("Retrieving")
    documents = retrieving.get_relevant_documents(question)
    return documents


def context_formatting(documents):
    """
    Formats retrieved documents to be used as context for the LLM
    :param documents: retrieved documents
    :return: formatted documents
    """
    content = ""
    for index, document in enumerate(documents):
        content = content + "[doc" + str(index + 1) + "]=" + document.page_content.replace("\n", " ") + "\n\n"
    return content


def source_formatting(documents):
    """
    Formats retrieved documents to be used as sources for the user
    :param documents: retrieved documents
    :return: formatted documents
    """
    sources = ""
    for index, document in enumerate(documents):
        sources = sources + "[doc" + str(index + 1) + "]=" + document.metadata["source"] + "\n\n"
    return sources.strip()


def generate(question, documents, use_llm):
    """
    LLM generates a response based on the question (user query), added context (retrieved documents), and a prompt
    :param question: user query
    :param documents: retrieved documents, formatted
    :param use_llm: which llm to use
    :return: LLM generated response
    """
    print("Generating")
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
    # define LLM to be used and the temperature (creativity/randomness) of the model
    llm = ChatOllama(model=use_llm, temperature=0)

    # define a LangChain chain
    chain = rag_prompt | llm | StrOutputParser()

    # invoke chain with retrieved documents and the question (user query)
    output = chain.invoke({"context": documents, "question": question})

    return output


if __name__ == "__main__":
    """
    main function
    """
    print("Starting program")

    # define what LLM to use
    use_llm = "mistral:instruct"

    # define what embedding model to use
    embedding = GPT4AllEmbeddings()

    # directory of the data files to do RAG on
    # change to ../text if you want to use .txt files stored in the 'text' folder
    # change to ../pdf if you want to use .pdf files stored in the 'pdf' folder
    data_directory = '../text'

    # directory to persistently store the vector embedding store
    db_directory = '../chroma_db'

    # creating or fetching vector store
    retriever = None
    if not os.path.isdir(db_directory):
        retriever = create_vector_store(db_directory, chunking(data_directory), embedding)
    else:
        retriever = fetch_vector_store(db_directory, embedding)


    def complete_rag(question):
        """
        The process of retrieval augmented generation
        :param question: user query
        :return: sources and LLM ouput, generated using retrieved documents
        """
        docs = retrieve(retriever, question)
        output = generate(question, context_formatting(docs), use_llm)
        return source_formatting(docs), output


    # for web view of prompting
    # code below is copied from: https://www.youtube.com/watch?v=JEBDfGqrAUA (Project 2)
    with gr.Blocks(theme=Base(), title="Q&A on your data with RAG") as demo:
        gr.Markdown("# Q&A on your data with RAG")
        textbox = gr.Textbox(label="Question:")
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
        with gr.Column():
            output1 = gr.Textbox(lines=1, max_lines=10, label="Sources")
            output2 = gr.Textbox(lines=1, max_lines=10,
                                 label="Generated output by LLM, incorporating retrieved documents")

        button.click(complete_rag, textbox, outputs=[output1, output2])

    demo.launch()