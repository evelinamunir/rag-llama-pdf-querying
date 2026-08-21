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

DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
MODEL_DIR = "./models"
MODEL_PATH = "./models/llama-3.2.gguf"


def clean_response(text: str) -> str:
    text = text.strip()

    for prefix in ["Answer:", "The final answer is:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    if "\nNote:" in text:
        text = text.split("\nNote:")[0].strip()
    elif "\n\nNote" in text:
        text = text.split("\n\nNote")[0].strip()

    return text


def main():
    print("Initialising RAG PDF Query System")

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created directory '{DOCS_DIR}'.")
        print(
            "Please place your PDF files inside this folder and run the script again."
        )
        sys.exit()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory '{MODEL_DIR}'.")
        print(
            "Please download a Llama 3.2 GGUF model file, name it 'llama-3.2.gguf', place it in the models folder, and run again."
        )
        sys.exit()

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at '{MODEL_PATH}'.")
        print(
            "Please download a Llama 3.2 GGUF file and place it in the models folder."
        )
        sys.exit()

    print(f"Looking in '{DOCS_DIR}' for PDFs.")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()

    if not docs:
        print(f"No PDFs were found in '{DOCS_DIR}'. Please add some and run again.")
        sys.exit()

    print(f"Successfully loaded {len(docs)} document pages.")

    print("Chunking documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    print(
        "Generating embeddings (on the first run, this will download the embedding model)."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=DB_DIR
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("Starting Llama model from local file.")
    llm = LlamaCpp(model_path=MODEL_PATH, temperature=0, n_ctx=2048, verbose=False)

    system_prompt = (
        "You are a strict data-extraction assistant. "
        "Extract the direct answer to the user's question using ONLY the provided context. "
        "Provide the final answer immediately. "
        "CRITICAL RULE: Do not include notes, evaluations, self-corrections, or explanations.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n")
    print("You can now ask questions about the documents.")
    print("Type 'exit' or 'quit' to close the app.")
    print("\n")

    while True:
        try:
            query = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye :)")
            break

        if query.lower() in ["exit", "quit"]:
            print("Goodbye :)")
            break

        if not query.strip():
            continue

        print("\n Model is generating an answer.")

        response = rag_chain.invoke({"input": query})

        cleaned_answer = clean_response(response["answer"])
        print("\nModel:", cleaned_answer)
        print()


if __name__ == "__main__":
    main()
