# DokScope RAG — Ask & Extract Insights from Your Documents

A local, privacy‑friendly Retrieval‑Augmented Generation (RAG) app built with **Streamlit**, **LangChain**, **ChromaDB**, and a local **Llama 3** (GGUF via `llama-cpp-python`).  
It performs OCR on PDFs (via **Mistral OCR** API), splits and enriches text, extracts Markdown tables to structured data, indexes everything, and answers questions grounded **only** in the retrieved context.

---

## ✨ Features

- **PDF → OCR → Markdown** with Mistral OCR (multilingual, layout aware).
- **Robust chunking** (Markdown‑aware + recursive character splitter with overlap).
- **Table extraction**: detects Markdown tables, stores them as CSV, and indexes a textual summary for retrieval.
- **Prompting compatible with Llama 3 chat format** (`system → user → assistant` headers).
- **Answer cleaning**: removes HTML/tags and deduplicates repeated paragraphs.
- **Configurable retriever** using **MMR (Maximal Marginal Relevance)** to balance relevance & diversity.
- **Source selection UI**: select/deselect all documents, delete selected, and keep the vectorstore in sync.
- **Local‑first**: embeddings + vectorstore persisted on disk; your files remain local.

---

## 🧱 Architecture

```
PDF → Mistral OCR → Markdown
            │
            ├─► Extract tables (regex) → CSV in ./data/table_data
            │                          → LangChain Document with table “summary”
            │
            ├─► Split text (Markdown headers + recursive) → enriched chunks
            │
            ├─► Embeddings (HF Sentence Transformers)
            │
            └─► Vector store (ChromaDB, persisted on disk)

Query → Retriever (MMR) → Build Llama‑3 prompt (system/user/assistant) → LLM
                                                       │
                                                       └─► Clean output (strip tags, de‑dupe)
```

---

## 📦 Requirements

- macOS (Apple Silicon **M1/M2/M3** supported) or Linux/Windows
- Python **3.11** (recommended)
- Conda (or Miniconda) for a clean virtual environment

---

## 📁 Project Structure (recommended)

```
rag/
├─ app.py                  # Streamlit app (main entry point)
├─ requirements.txt        # Python dependencies
├─ .env                    # Local secrets (MISTRAL_API_KEY=...)
└─ data/
   ├─ models/              # Local GGUF models (e.g., Llama-3.2-3B-Instruct.Q4_K_M.gguf)
   ├─ db/                  # Chroma vectorstore
   ├─ table_data/          # CSVs generated from Markdown tables
   └─ temp/                # Temporary files (uploaded PDFs, OCR text, etc.)
```

> Paths are created automatically by `setup_directories()` in `app.py`.  
> You can change the base directory by editing that helper.

---

## 🛠️ Setup

### 1) Create and activate a Conda env

```bash
conda create -n rag-env python=3.11 -y
conda activate rag-env
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> On Apple Silicon, `llama-cpp-python` uses Metal (no CUDA required). If you hit build issues:
>
> ```bash
> CMAKE_ARGS="-DLLAMA_METAL=on" pip install --no-cache-dir --force-reinstall llama-cpp-python
> ```

### 3) Download a local Llama 3 (GGUF)

- Place a model file (e.g., `Llama-3.2-3B-Instruct.Q4_K_M.gguf`) under `./data/models/`.
- In `app.py`, set:
  ```python
  llama_model_path = "./data/models/Llama-3.2-3B-Instruct.Q4_K_M.gguf"
  ```
- The app will raise a `FileNotFoundError` if the model is missing.

### 4) Configure environment variables

Create a `.env` file at the project root:

```dotenv
MISTRAL_API_KEY=your_api_key_here
```

> **Important:** `.env` is listed in `.gitignore` so it won’t be committed.  
> At runtime the app also provides a **sidebar field** to enter a key manually (useful for demos).

---

## ▶️ Run

```bash
conda activate rag-env
streamlit run app.py
```

Open the local URL printed by Streamlit.

---

## 📚 Usage

1. **Upload PDF(s)** via the drag‑and‑drop box in the main page.
2. Click **Browse files**:
   - The app runs OCR → extracts Markdown → replaces tables with `TABLE_ID` references.
   - Tables are saved as CSV in `./data/table_data/` and a textual summary is indexed.
   - Chunks and table “documents” are embedded and stored in ChromaDB.
3. **Ask a question** in the chat box.
4. **Select sources**:
   - Use **“Select All Documents”** to toggle all docs.
   - Uncheck individual documents to restrict retrieval to a subset.
5. **Delete**:
   - Use **"Delete Selected"** to remove selected docs, their temp files, and their records in the vectorstore.
   - Associated CSVs in `table_data/` with the document prefix are also removed.

---

## 🧠 LLM Prompting (Llama 3 chat format)

The app builds prompts using the Llama 3 header format:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
CONTEXT:
{context}

QUESTION:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

**System prompt rules (summary):**

- Plain-text responses in the same language as the input question (no HTML/Markdown/tags)
- Don't repeat content; quote numbers/dates **exactly** as in the context
- If the answer isn't in the context:  
  `I don't have enough information to answer this question.`

---

## 🧩 Key Components

### OCR

- `process_pdf_with_mistral_ocr(pdf_path_or_file)`
  - Returns Markdown and writes a `.txt` copy in `temp/`.
  - Errors: raises or logs and returns `(None, None)` depending on context.

### Table extraction

- `extract_tables_from_markdown(markdown_text)`

  - Uses a regex to capture header + rows, replaces tables with references like `[TABLE_REFERENCE: TABLE_0]`.
  - Returns `(modified_text, extracted_tables)` where each table has:
    ```json
    {
      "id": "TABLE_0",
      "content": "...markdown...",
      "description": "Table 1 extracted from document"
    }
    ```

- `create_table_document(table_data)`
  - Builds a LangChain `Document` summarizing the table:
    ```
    Table ID: TABLE_0
    Description: Table 1 extracted from document
    Columns: ...
    Number of rows: N
    Table content:
    <df.to_string()>
    ```
  - Metadata includes `{"source": document_name, "table_id": ..., "description": ...}`.

### Chunking & indexing

- Markdown header splitter (`MarkdownHeaderTextSplitter`) + `RecursiveCharacterTextSplitter`:
  - `chunk_size=1000`, `chunk_overlap=200`, ordered separators.
- Enrich chunks with header context and optional de‑duplication.
- Persist via Chroma:
  ```python
  Chroma(persist_directory=dirs["db_dir"], embedding_function=self.embeddings)
  ```

### Retrieval

- MMR retriever:
  ```python
  retriever = vectorstore.as_retriever(
      search_type="mmr",
      search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.7}
  )
  ```
- `query(question)`:
  - Either uses `RetrievalQA` with a custom prompt **or**
  - Manually fetches `k` docs, builds the Llama‑3 prompt, invokes the model, then cleans the answer.

### Cleaning utilities

- `clean_llm_response(text)`:

  - Strip HTML/tags
  - Drop "QUESTION:" / "IMPORTANT:" blocks if present
  - De-duplicate paragraphs

- `dedup_paragraphs(text)`:
  - Remove repeated adjacent paragraphs while preserving order

### Logging

- Standard Python logging with `logger = logging.getLogger(__name__)`.
- (Optional) Streamlit sidebar shows logs via an in‑memory buffer (`io.StringIO`).

---

## 🧹 Deleting Documents & Tables

When you click **Remove Files** the app:

1. Deletes the corresponding entries from the vectorstore, e.g.:
   ```python
   self.vectorstore._collection.delete(where={"source": doc_name})
   ```
2. Removes the temp PDF and OCR text from `data/temp/`.
3. Removes **CSV tables** starting with the document prefix from `data/table_data/`.
4. Updates Streamlit state and, if no docs remain, resets the pipeline and chat.

> This ensures no orphaned tables/embeddings remain after removing a document.

---

## 🧯 Troubleshooting

- **`FileNotFoundError: LLM model not found ...`**  
  Ensure `llama_model_path` points to an existing GGUF file under `./data/models/`.

- **`NotImplementedError: This client is deprecated ...` (Mistral)**  
  Pin `mistralai==0.4.2` _or_ migrate to the new client per the official migration guide.  
  The repo uses the **new** client API by default.

- **Streamlit asks for your email at startup**  
  Just press **Enter** to skip.

- **Widget state errors (e.g., “cannot be modified after instantiation”)**  
  Initialize `st.session_state.select_all_sources` **before** you render the checkbox and avoid setting it in the same render after the widget is created.

- **Uploaded file re‑processing loops**  
  Keep a `processed_files` set in `st.session_state` and skip files already processed in the current session.

- **Large files**  
  Streamlit’s uploader defaults to ~200 MB. You can increase using:
  ```bash
  export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024
  ```
  then restart Streamlit. Consider memory constraints.

---

## 🔐 Privacy & Security

- All processing runs locally except OCR calls to Mistral.
- You control which documents are indexed and which sources are used to answer.
- `.env` is excluded from git; never commit secrets.

---

## 🗺️ Roadmap

- Multi‑file table cross‑references
- Inline table rendering in answers
- Better hallucination guards (answerability classifier)
- Optional cloud vectorstore
- Eval suite for retrieval quality

---

## 📝 License

MIT — see `LICENSE`.

---

## 🙌 Acknowledgements

- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) & `llama-cpp-python`
- Hugging Face sentence transformers
- Mistral OCR API
