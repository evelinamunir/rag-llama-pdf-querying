# Local RAG PDF Query System

This is a local Retrieval-Augmented Generation (RAG) system that allows you to query PDF documents using a locally hosted Llama 3.2 model.

## Setup Instructions

1. **Install Dependencies**
   Run the following command to install the required Python packages:
   `pip install -r requirements.txt`

2. **Download the Model**
   * Download a Llama 3.2 GGUF model file.
   * Run the script once; it will automatically create a `models/` directory (or make it yourself).
   * Place your downloaded model into the `models/` folder and rename it exactly to `llama-3.2.gguf`.

3. **Add Your PDFs**
   * Run the script again; it will generate a `docs/` folder (or make it yourself).
   * Place the PDF files you want to query inside the `docs/` folder.

4. **Run the Application**
   Run the `Query_PDFs.py` script. On the first successful run, it will chunk the PDF documents, download the Nomic embedding model, and build a local Chroma database in a `chroma_db/` folder. Once loaded, type your questions into the terminal.


   ## Important Notes
This runs on your CPU, but can be adapted to use a graphics card instead if wanted.

* **The First Run:** When you run the script for the first time, it will automatically download the Nomic embedding model from Hugging Face. This may take a few minutes depending on your internet connection. 
* **Security Note:** The Hugging Face embeddings use `trust_remote_code=True`. Ensure you are comfortable with this before running.

## Troubleshooting

**Missing C++ Build Tools (Windows)**
If you get a wall of red text when running `pip install -r requirements.txt` (specifically failing on `llama-cpp-python` or `orjson`), you need to install the Microsoft C++ Build Tools. Download the installer from Microsoft, tick the box for "Desktop development with C++", install it, and try again.