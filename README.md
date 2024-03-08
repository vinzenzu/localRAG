# localRAG

## Fully free, local and open-source Q&A RAG with Mistral 7B, using local documents


#### Set up:

Tested using Python 3.12.    
Reproducable steps to run this code:  
0. Along the way, you may be asked to install Microsoft C++ Build Tools. You can do so from here: https://visualstudio.microsoft.com/visual-cpp-build-tools/
1. Download or clone this repository.
2. Install Ollama.
3. In the terminal (e.g. PowerShell), run ```ollama pull mistral:instruct``` (or pull a different model of your liking, but make sure to change the variable use_llm in the Python code accordingly)
4. Set up a new Python virtual environment. For best convenience, use an IDE like PyCharm for this.
5. To install the necessary packages to the venv (virtual environment), run ```pip install gradio langchain gpt4all chromadb pypdf tiktoken``` in the terminal of the venv. If you don't want status updates printed to your terminal, run it with the quiet flag: ```pip install --quiet gradio langchain gpt4all chromadb pypdf tiktoken```. You can run this command in the Python terminal in PyCharm, for example, by clicking on the terminal button in the bottom left column of the interface. This command should grab all necessary dependencies to run the code. If, for an unknown reason, running the code (step 8) gives an error, and it prompts you to install a Python package to be able to run the code, please do so.
6. Add either your pdf files to the pdf folder, or add your txt files to the text folder. Change the data_directory in the Python code according to which data you want to use for RAG. (If you ever want to switch, change, and modify which data you use for RAG, simply delete the chroma_db folder that gets created when the data is prepared for RAG during the first run of the program and re-run the program to create a new vector database.)
7. Start Ollama.
8. Run the python file. The first run may take a while.
9. In the terminal, a local IP address will be printed. Copy it, paste it into a browser, and you can interact with your documents with RAG using a LLM. For the process of asking questions, see below.
