# Imports
import os
import sys

from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Config constants
DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
MODEL_DIR = "./models"
MODEL_PATH = "./models/llama-3.2.gguf"


def clean_response(text: str) -> str:
    """Cleans the response by removing prefixes and notes."""
    text = text.strip()

    # Removes standard prefixes generated
    for prefix in ["Answer:", "The final answer is:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    # Removes any notes that might be appended to the answer
    if "\nNote:" in text:
        text = text.split("\nNote:")[0].strip()
    elif "\n\nNote" in text:
        text = text.split("\n\nNote")[0].strip()

    return text


def main():
    print("Initialising RAG PDF Query System")

    # Environment Setup and Validation
    # Checks if the docs directory exists, if not, creates it and prompts the user to add PDFs
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created directory '{DOCS_DIR}'.")
        print(
            "Please place your PDF files inside this folder and run the script again."
        )
        sys.exit()

    # Checks if the models directory exists, if not, creates it and prompts the user to add the Llama model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory '{MODEL_DIR}'.")
        print(
            "Please download a Llama 3.2 GGUF model file, name it 'llama-3.2.gguf', place it in the models folder, and run again."
        )
        sys.exit()

    # Checks if the Llama model file exists, if not, prompts the user to download it and put it in the models folder
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at '{MODEL_PATH}'.")
        print(
            "Please download a Llama 3.2 GGUF file and place it in the models folder."
        )
        sys.exit()

    # Document Loading and Processing
    print(f"Looking in '{DOCS_DIR}' for PDFs.")
    # Initialises the PDF loader for the directory
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    # Extracts text from the PDFs
    docs = loader.load()

    if not docs:
        print(f"No PDFs were found in '{DOCS_DIR}'. Please add some and run again.")
        sys.exit()

    print(f"Successfully loaded {len(docs)} document pages.")

    # Document Chunking
    print("Chunking documents.")
    # Splits the documents into 800 character chunks with 100 character overlap to ensure context is preserved
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Embedding Generation
    print(
        "Generating embeddings (on the first run, this will download the embedding model)."
    )

    # Converts text into numerical vectors so semantic meaning can be understood
    # Uses Nomic model from HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Vector Database Setup
    # Stores the chunked documents and vector embeddings into a Chroma database
    # Saves to disk to avoid re-embedding on subsequent runs
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=DB_DIR
    )

    # Sets up a retriever to fetch the top 4 most relevant document chunks based on the user's query
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LLM Initialisation
    print("Starting Llama model from local file.")

    # Loads the local Llama model, temperature is set to 0 for deterministic output, context window is max 2048 tokens
    llm = LlamaCpp(model_path=MODEL_PATH, temperature=0, n_ctx=2048, verbose=False)

    # Prompting Instructions
    system_prompt = (
        "You are a strict data-extraction assistant. "
        "Extract the direct answer to the user's question using ONLY the provided context. "
        "Provide the final answer immediately. "
        "CRITICAL RULE: Do not include notes, evaluations, self-corrections, or explanations.\n\n"
        "Context:\n{context}"
    )

    # Combines the system instructions with the user's input to form a complete prompt for the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Building RAG Pipeline
    # Creates a chain that first retrieves relevant chunks and puts them in the 'context' variable of the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Combines the retriever and document chain into the final pipeline
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n")
    print("You can now ask questions about the documents.")
    print("Type 'exit' or 'quit' to close the app.")
    print("\n")

    # Interactive Query Loop
    while True:
        try:
            # Get user input for the query
            query = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            # Handles Ctrl+C
            print("\nGoodbye :)")
            break

        # Checks for exit commands
        if query.lower() in ["exit", "quit"]:
            print("Goodbye :)")
            break

        # Ignore empty inputs
        if not query.strip():
            continue

        print("\n Model is generating an answer.")

        # Passes the query to the RAG chain
        response = rag_chain.invoke({"input": query})

        # Cleans the output and prints it to the console
        cleaned_answer = clean_response(response["answer"])
        print("\nModel:", cleaned_answer)
        print()


if __name__ == "__main__":
    main()
