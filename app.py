import os
import base64
import re
import io
#import json
import time
import tempfile
#import uuid # generate unique IDs
import logging
import traceback
import pandas as pd
import markdown
import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
#from langchain.retrievers.document_compressors import EmbeddingsFilter
#from langchain.retrievers import ContextualCompressionRetriever
#from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.llms import LlamaCpp
#from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain_community.retrievers import BM25Retriever
#from langchain_community.retrievers.merger_retriever import MergerRetriever
#from langchain.retrievers.document_compressors import LLMChainExtractor
#from langchain_core.retrievers import MultiVectorRetriever
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
#from langchain_core.prompts import ChatPromptTemplate
#from langchain.retrievers.multi_query import MultiQueryRetriever
#from langchain.retrievers import ContextualCompressionRetriever
#from langchain.retrievers import ParentDocumentRetriever
#from langchain.storage import InMemoryStore
#from langchain.retrievers import MMR

#import mistralai
#from mistralai.client import MistralClient
from mistralai import Mistral
from mistralai.client import MistralClient

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API key configuration
def get_api_key(key_name: str) -> str:
    """Get an API key from environment variables or from Streamlit session state"""
    api_key = os.environ.get(key_name, "")
    if not api_key:
        api_key = st.session_state.get(key_name, "")
    return api_key

# Create required directories
def setup_directories() -> Dict[str, str]:
    """Create the directories required by the application"""
    base_dir = Path("./data")
    temp_dir = base_dir / "temp"
    db_dir = base_dir / "db"
    model_dir = base_dir / "models"
    table_dir = base_dir / "table_data"
    
    # Create directories if they don't exist
    for directory in [base_dir, temp_dir, db_dir, model_dir, table_dir]:
        directory.mkdir(exist_ok=True, parents=True)
    
    return {
        "base_dir": str(base_dir),
        "temp_dir": str(temp_dir),
        "db_dir": str(db_dir),
        "model_dir": str(model_dir),
        "table_dir": str(table_dir)
    }

# Utility functions for markdown table extraction and conversion
def extract_tables_from_markdown(markdown_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract markdown tables from text and replace them with references"""
    # Pattern regex to identify markdown tables
    table_pattern = r'(\|[^\n]+\|\n)(\|[-:| ]+\|\n)((?:\|[^\n]*\|\n)+)'

    # (\|[^\n]+\|\n)
    # - Match a line that starts and ends with "|" (pipe)
    # - [^\n]+: any character except newline, at least once (i.e., header/cell content)
    # - \n: end of line
    # -> Corresponds to the table header row (e.g., | Col1 | Col2 |)

    # (\|[-:| ]+\|\n)
    # - Match the separator line following the header line
    # - [-:| ]+: one or more of "-", ":", "|" or space (markdown table syntax)
    # - \n: end of line
    # -> Corresponds to the column separator row (e.g., | --- | --- |)

    # ((?:\|[^\n]+\|\n?)+)
    # - (?:...): non‑capturing group to avoid unwanted subgroups
    # - \|[^\n]+\|: a line that starts and ends with "|" and has at least one non‑newline char (data cells)
    # - \n?: optional newline at the end of each data row
    # - +: one or more data rows
    # -> Corresponds to the table data rows

    # This pattern recognizes a standard markdown table with:
    # - One header line
    # - One separator line (---)
    # - One or more data lines
    
    # List that will contain dicts with info about each extracted table
    extracted_tables = []
    # Helper list to manage references that will replace tables in the text
    table_references = []

    # Find all table occurrences in markdown text
    table_matches = list(re.finditer(table_pattern, markdown_text))

    # Make a mutable copy of the text where tables will be progressively removed
    modified_text = markdown_text

    # Iterate over all found tables
    for i, match in enumerate(table_matches):
        table_text = match.group(0)  # Whole markdown table (header + separator + data rows)
        table_id = f"TABLE_{i}"      # Unique identifier for each table

        # String that will replace the table with its reference
        table_reference = f"\n[{table_id}: Tabella estratta dal documento]\n"

        # Store detailed table info into the extracted list
        extracted_tables.append({
            "id": table_id,              # unique id of the table (e.g., TABLE_0)
            "content": table_text,       # raw markdown of the table
            "description": f"Table {i+1} extracted from document"  # optional description
        })

        # Store reference info for replacement in text
        table_references.append({
            "id": table_id,              # table id
            "reference": table_reference,# string that will replace the table in the text
            "original": table_text       # original text to be replaced
        })

    # Replace original tables with references in the text,
    # processing backwards to avoid breaking positions of other matches
    for ref in reversed(table_references):
        modified_text = modified_text.replace(ref["original"], ref["reference"])

    # Return the modified text (without tables) and the list of extracted tables
    return modified_text, extracted_tables

def markdown_table_to_dataframe(table_markdown: str) -> Optional[pd.DataFrame]:
    """
    Convert a markdown table into a pandas DataFrame.
    Return None on error.
    """
    try:
        # Divide the table into lines
        lines = table_markdown.strip().split('\n')  # Strip whitespace and split by lines

        # Extract headers from the first line
        headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]  
        # .split('|') splits cells; [1:-1] drops leading/trailing empty cells caused by the split

        # Skip the second line (separator row like | --- | --- |)
        # Data = all lines from index 2 onward
        data = []
        for line in lines[2:]:
            if line.strip():  # Consider only non-empty lines
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                data.append(cells)

        # Create the DataFrame
        df = pd.DataFrame(data, columns=headers)
        return df
    except Exception as e:
        print(f"Error converting the markdown table to a Dataframe: {e}")
        return None
    
def create_table_document(table_data: Dict[str, Any], source_file: str) -> Optional[Document]:
    """
    Create a LangChain Document from a markdown table
    """
    # Convert the markdown table into a pandas DataFrame
    df = markdown_table_to_dataframe(table_data["content"])

    if df is not None:
        # Generate a detailed textual description of the table for indexing
        table_description = f"""
        ID Table: {table_data['id']}
        Description: {table_data.get('description', '')}
        Column: {', '.join(df.columns.tolist())}
        Rows Number: {len(df)}

        Content of the table:
        {df.to_string(index=False)}
        """

        # Create a LangChain Document with additional metadata
        doc = Document(
            page_content=table_description,
            metadata={
                "source": source_file,               # Source file
                "table_id": table_data["id"],        # Unique table ID
                "description": table_data.get('description', '')
            }
        )
        return doc

    # If conversion fails, return None (the table will not be indexed)
    return None

# Utility functions for text cleaning
def dedup_paragraphs(text: str) -> str:
    """
    Remove contiguous duplicate paragraphs from text.
    """
    
    # Split text into paragraphs, stripping whitespace and dropping empty ones
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    out = []
    for p in paras:
        if not out or p != out[-1]:
            out.append(p)
    # Recompose text joining paragraphs with two newlines
    return "\n\n".join(out)

def collapse_identical_lines(text: str) -> str:
    """Remove consecutive identical lines"""
    lines = text.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        if i == 0 or line.strip() != lines[i-1].strip():
            result.append(line)
    
    return '\n'.join(result)

def clean_llm_response(text: str) -> str:
    # Clean LLM output from HTML/tags, repetitions, and excessive whitespace
    text = re.sub(r'<.*?>', '', text)  # Remove basic HTML tags like <div>, <p>, <table>, etc.
    text = re.sub(r'QUESTION:.*', '', text)  # remove the question part
    text = re.sub(r'IMPORTANT:.*', '', text)  # remove the IMPORTANT section
    text = dedup_paragraphs(text)  # remove contiguous duplicates
    return text.strip()

# Prompt system
prompt_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant that answers ONLY based on the context below, composed of one or more excerpts from documents (including tables or structured data).

RULES:
- Provide the answer as plain text without any HTML, markdown, or special tags.
- Do NOT repeat the same information/sentences more than once.
- If the answer is a number, date, percentage, or a specific datum, report the value EXACTLY as it appears in the context and indicate where it comes from (e.g., "According to the document: ...").
- If the datum is in a table, explicitly cite the value, the row and column, and copy the exact sentence/row containing it.
- Do NOT add the sentence "I don't have enough information" if you have already partially answered.
- If the answer is not present in the context, reply only: I don't have enough information to answer this question.
- Do NOT repeat the question in your answer, and do NOT add generic explanations or greetings.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
CONTEXT:
{context}

QUESTION:
{question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def build_prompt(context: str, question: str) -> str:
    return prompt_template.format(context=context, question=question)

# Main RAG Pipeline Class
class RagPipeline:
    def __init__(self, temperature=0.0000001):
        # --- Embedding Model ---
        # elect device: use CUDA if available, then MPS (Apple Silicon), otherwise CPU
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        self.dirs = setup_directories()
        self.mistral_client = None

        # Configure the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # multilingual embeddings
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True} #L2-normalize embeddings to unit length. Normalization removes magnitude effects and focuses on direction (better semantic similarity). CThis means that two semantically similar texts will have similar embedding directions regardless of their length or the specific word frequencies
        )

        # --- Vector Store ---
        # Initialize vector store
        dirs = setup_directories()
        self.vectorstore = Chroma(
            persist_directory=dirs["db_dir"], # directory where embeddings are persisted
            embedding_function=self.embeddings # link the embedding model to the vectorstore
        )

        # --- Text Splitter ---
        # General-purpose character splitter with overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, # overlap between consecutive chunks (in characters)
            separators=["\n\n", "\n", ".", " ", ""],  # separator priority: double newline, newline, period, space, fallback
            length_function=len # Python built-in len() function to measure text length
        )

        # Splits markdown by headers (useful for structured documents)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3")
            ]
        )

        # --- LLM Configuration (LlamaCpp) ---
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])  # Stream model output to the console (stdout) (handy for real-time debug)

        # Local path to the model
        llama_model_path = "./data/models/Llama-3.2-3B-Instruct.Q4_K_M.gguf"
        if not os.path.exists(llama_model_path):
            raise FileNotFoundError(
                f"LLM Model not found in {llama_model_path}."
            )
        
        self.llm = LlamaCpp(
            model_path=llama_model_path,
            temperature=temperature,
            max_tokens=2048,
            n_ctx=4096,
            n_gpu_layers=-1,  # Use all available GPU layers (works for MPS too
            n_batch=512, # Larger values improve throughput (more tokens per batch) at the cost of memory
            callbacks=callback_manager,
            verbose=False,
            f16_kv=True,    # Use float16 for key/value cache (saves memory, may be slightly less accurate)
            use_mlock=True  # Block model memory from being swapped to disk (improves performance)
        )

        # Retrieval/QA chain Initialization
        self.qa_chain = None # Initialize retriever after documents are loaded
    
    def process_pdf_with_mistral_ocr(self, pdf_path):
        """Process a PDF with Mistral OCR and return extracted text and the text file path"""
        try:
            # Load file 
            uploaded_pdf = self.mistral_client.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": open(pdf_path, "rb"), # the file is opened in binary mode (“rb”) to allow reading of files such as images or other non-text files.
                },
                purpose="ocr" # specifies that the file will be used for OCR
            )

            # Get the signed URL to access the file
            signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)

            # Run OCR on the document
            ocr_response = self.mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url", # indicates that the document is provided via a URL 
                    "document_url": signed_url.url, # provides the signed URL of the uploaded PDF
                },
                include_image_base64=True # requires that the response include images in base64 format
            )

            # Extract the markdown content from all pages
            markdown_content = "" #initialize empty string
            for page in ocr_response.pages:
                markdown_content += page.markdown # add each page's markdown to the full content

            # Save the OCR result as a txt for processing
            txt_path = os.path.join(self.dirs["temp_dir"], f"{os.path.basename(pdf_path)}.txt") # make a path for the output file in the temp directory
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(markdown_content) # write the extracted markdown content to the text file

            logger.info(f"OCR completed for {pdf_path}. Output saved in {txt_path}")
            return markdown_content, txt_path  # Returns both the text and the file path
        except Exception as e:
            logger.error(f"Error during OCR processing of {pdf_path}: {str(e)}")
            return None, None


    def convert_markdown_table_to_dataframe(table_markdown: str) -> Optional[pd.DataFrame]:
        """
        Converts the markdown table into a pandas DataFrame. Return None on error.
        """
        try:
            # Divide the table into lines
            lines = table_markdown.strip().split('\n') # strip() removes leading/trailing whitespace and split('\n') splits the text into a list of lines.

            # Estract headers from the first line
            headers = [cell.strip() for cell in lines[0].split('|')[1:-1]] 
            # Example:
            # "| Col1 | Col2 |".split('|')   -> ["", " Col1 ", " Col2 ", ""]
            # [1:-1]                         -> [" Col1 ", " Col2 "]
            # strip()                        -> ["Col1", "Col2"]
            # Ignore the separator line (second line)
 
            # Estract data from all lines starting from the third
            data = []
            for line in lines[2:]:
                if line.strip():
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    data.append(cells)

            # Create the DataFrame
            df = pd.DataFrame(data, columns=headers)
            return df
        except Exception as e:
            print(f"Error converting markdown table to DataFrame: {e}")
            return None


    def process_and_index_document(self, text, document_name):
        """
        Process the document text, extract tables, and index everything into the vector store
        """
        # 1. Exrtact tables from markdown
        modified_text, extracted_tables = extract_tables_from_markdown(text)  # Removes tables from the text and saves them separately

        # 2. Create base metadata for the document
        base_metadata = {"source": document_name}  # Metadata will be added to each chunk

        # 3. Process the main text (without tables)
        # First, split by markdown headers
        markdown_docs = self.markdown_splitter.split_text(modified_text)

        # Then, apply normal chunking on the split documents
        text_chunks = []
        for doc in markdown_docs:  # docs already split by headers
            # Combine the content with any found headers
            content = doc.page_content
            header_info = ""
            if "header1" in doc.metadata:
                header_info += f"# {doc.metadata['header1']}\n"
            if "header2" in doc.metadata:
                header_info += f"## {doc.metadata['header2']}\n"
            if "header3" in doc.metadata:
                header_info += f"### {doc.metadata['header3']}\n"

            # If headers exist, prepend them (trimmed) to the content with a blank line and strip leading spaces from content; otherwise keep content as is.
            content_with_headers = f"{header_info.strip()}\n\n{content.lstrip()}" if header_info else content

            # Chunking the document
            chunks = self.text_splitter.split_text(content_with_headers)

            # Add header metadata to each text chunk
            for chunk in chunks:
                chunk_metadata = base_metadata.copy()  # Paste metadata base
                if "header1" in doc.metadata:
                    chunk_metadata["header1"] = doc.metadata["header1"]
                if "header2" in doc.metadata:
                    chunk_metadata["header2"] = doc.metadata["header2"]
                if "header3" in doc.metadata:
                    chunk_metadata["header3"] = doc.metadata["header3"]
                text_chunks.append((chunk, chunk_metadata))  # Save chunk and metadata

        # 4. Process the extracted tables
        table_documents = []
        for table_data in extracted_tables:
            # Create the LangChain Document from the markdown table
            table_doc = create_table_document(table_data, document_name)
            if table_doc:
                # Table metadata: base + specific table metadata
                table_metadata = base_metadata.copy()
                table_metadata.update(table_doc.metadata)
                table_documents.append((table_doc.page_content, table_metadata))

                # Save the table as CSV for future reference
                df = RagPipeline.convert_markdown_table_to_dataframe(table_data["content"])
                if df is not None:
                    csv_path = os.path.join(self.dirs["table_dir"], f"{document_name}_{table_data['id']}.csv")
                    df.to_csv(csv_path, index=False)

        # 5.e Combined documents to index
        all_documents = text_chunks + table_documents

        # 6. Add everything to the vectorstore (texts and metadata)
        texts = [doc[0] for doc in all_documents]
        metadatas = [doc[1] for doc in all_documents]
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        self.vectorstore.persist()  #  Save changes to disk

        print(f"Indexed {len(texts)} chunks to the vector store (included {len(table_documents)} tables).")
        return len(texts), len(table_documents)

    def setup_retrieval_chain(self):
        """Configures the retrieval chain for QA."""
        retriever = self.vectorstore.as_retriever( # Retriever object created from the vectorstore. The as_retriever() method converts the vector database into a retrieval system that can be queried to find relevant documents.
            search_type="mmr",  # Maximal Marginal Relevance (MMR) for diversity in results. This algorithm does not simply return the most similar documents to the query, but tries to balance relevance with diversity, avoiding returning documents that are too similar to each other.
            search_kwargs={
                "k": 6,  # top-k results: return up to 6 most relevant & diverse chunks (post‑MMR)
                "fetch_k": 12, # candidate pool size before MMR re-ranking
                "lambda_mult": 0.7  # trade‑off relevance vs diversity (0.7 favors relevance)
            }
        )


        # The "lambda_mult" parameter controls the balance between relevance and diversity in the MMR algorithm:

        # - Values close to 1.0 favor relevance (documents more similar to the query)
        # - Values close to 0.0 favor diversity (documents more different from each other)
        # - 0.7 is an intermediate value that gives more weight to relevance but also considers diversity



        self.qa_chain = RetrievalQA.from_chain_type( # Initialize the qa_chain attribute of the current object using the static method from_chain_type of the RetrievalQA class. This method creates a Retrieval-based Question Answering chain.
            llm=self.llm, # Specificies the language model (LLM) to use for generating answers.
            chain_type="stuff",  # Useful for shorter documents
            retriever=retriever,
            return_source_documents=True,  # Useful to show the source documents used to generate the answer, allowing to see the sources.
        )


        # - chain_type="stuff":
        # dump all retrieved chunks into one prompt (simple/fast; good for short contexts;
        # This approach is effective when the retrieved documents are relatively short and can all fit into the model's context. If the documents are too long, they may need to be truncated or summarized to fit.



    def query(self, question: str, selected_docs: List[str] = None) -> Dict[str, Any]:
        """Run a query over the RAG system, optionally filtering by selected documents"""
        if not self.qa_chain:
            self.setup_retrieval_chain()
        
        logger.info(f"Running query: {question}")

        try:
            # Retrieval with filter for selected documents
            if selected_docs:
                docs = self.vectorstore.similarity_search(
                    question, 
                    k=6,
                    filter={"source": {"$in": selected_docs}}
                )
            else:
                # If no documents are selected, use all documents
                docs = self.vectorstore.similarity_search(question, k=6)
                
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = build_prompt(context, question)
            answer = self.llm.invoke(prompt)
            cleaned_answer = clean_llm_response(answer)

            return {
                "answer": cleaned_answer,
                "source_documents": docs
            }
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {
                "answer": "An error occurred while processing the query.",
                "source_documents": []
            }

    def cleanup_temp_files(self):
        """Clean up ALL temporary files, keeping only those of the current document"""
        try:
            # Get the path of the temporary directory from the dirs initialized in the constructor
            temp_dir = Path(self.dirs["temp_dir"])
            
            # Initialize an empty set to store the names of files to keep
            current_files = set()

            # Checks if there are uploaded files in the current session
            if hasattr(st.session_state, 'uploaded_files'):
                # Iterate over all documents uploaded in the current session
                for doc_info in st.session_state.uploaded_files.values():
                    # Extract the file name from the full path
                    base_name = Path(doc_info["path"]).name
                    # Add the original PDF file name to the list of files to keep
                    current_files.add(base_name)  
                    # Add the corresponding .txt file generated by OCR
                    current_files.add(f"{base_name}.txt")  
            
            # Scan all files in the temporary directory
            for file_path in temp_dir.glob("*"):
                # If the file is not in the list of files to keep
                if file_path.name not in current_files:
                    try:
                        # Remove the file not needed
                        file_path.unlink()
                        # Log the removal operation
                        logger.info(f"Temporary file removed: {file_path}") 
                    except Exception as e:
                        # If there's an error removing a single file, it logs it but continues
                        logger.error(f"Error removing {file_path}: {e}")
                        
        except Exception as e:
            # Log error during cleanup
            logger.error(f"Error in cleaning temporary files: {e}")

    def load_existing_documents(self):
        """Load processed documents from the vector database"""
        try:
            db_path = Path(self.dirs["db_dir"])
            if db_path.exists():
                # Vectorstore already initialized in constructor
                docs = self.vectorstore.get()
                if docs and docs['metadatas']:
                    unique_docs = {}
                    # Create a dictionary of unique documents
                    for meta in docs['metadatas']:
                        source = meta['source']
                        if source not in unique_docs:
                            unique_docs[source] = {
                                "path": os.path.join(self.dirs["temp_dir"], source),
                                "total_chunks": sum(1 for m in docs['metadatas'] if m['source'] == source),
                                "table_count": sum(1 for m in docs['metadatas'] if m['source'] == source and 'table_id' in m)
                            }
                    return unique_docs
            return {}
        except Exception as e:
            logger.error(f"Error during loading documents: {e}")
            return {}
        
    def remove_documents(self, document_names: List[str]):
        """Remove documents and embeddings from the vectorstore with relatives table CSVs"""
        try:
            # 1) Remove embeddings from the vectorstore using a filter for 'source' (more robust than deleting by id)
            for doc_name in document_names:
                try:
                    self.vectorstore._collection.delete(where={"source": doc_name})
                except Exception as e:
                    logger.warning(f"Unable to delete from Chroma the document '{doc_name}' via where: {e}") 
            self.vectorstore.persist()

            # 2) Delete CSVs of linked tables
            table_dir = Path(self.dirs["table_dir"])
            for doc_name in document_names:
                # Files are saved as: "<document_name>_TABLE_<n>.csv"
                pattern = f"{doc_name}_TABLE_*.csv"
                for csv_path in table_dir.glob(pattern):
                    try:
                        csv_path.unlink()
                        logger.info(f"Table removed: {csv_path.name}")
                    except Exception as e:
                        logger.warning(f"Unable to remove {csv_path}: {e}")

            logger.info(f"Embeddindgs and tables removed for: {document_names}")

        except Exception as e:
            logger.error(f"Error in removing documents: {e}")


# --- STREAMLIT APPLICATION ---
def create_streamlit_app():
    """Create the Streamlit application"""
    # Page config
    st.set_page_config(
        page_title="DocSkope",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize an in-memory log buffer to display logs directly in the app
    # if 'log_output' not in st.session_state:
    #     st.session_state.log_output = io.StringIO()
    
    # Logger config
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Initialize session state
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RagPipeline()
        # Clean up temporary files at session startup
        st.session_state.rag_pipeline.cleanup_temp_files()

    if "uploaded_files" not in st.session_state:
        # Load existing documents from vectorstore
        existing_docs = st.session_state.rag_pipeline.load_existing_documents()
        st.session_state.uploaded_files = existing_docs
        if existing_docs:  # If there are documents, initialize the chain
            st.session_state.rag_pipeline.setup_retrieval_chain()
    # Initialize chat history in Streamlit's session_state on first run (persists across reruns)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Ensure a session-persistent registry of already-processed file names.
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()   
    
    # Application title
    st.title("📚 DocSkope - Ask documents. Get ansewers")
    st.markdown("Load and chat with your PDF documents.")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Manage your Documents")
        
        # API Mistral API Mistral OCR config 
        mistral_api_key = st.text_input(
            "🔑 Enter your Mistral API key", 
            value=get_api_key("MISTRAL_API_KEY"),
            type="password",
            help="The key will never be stored and you can change it at any time"
        )
        if not mistral_api_key:
            st.warning("⚠️ Enter your Mistral API key to enable PDF OCR.")
            st.stop()
        
        if mistral_api_key:
            st.session_state["MISTRAL_API_KEY"] = mistral_api_key
            os.environ["MISTRAL_API_KEY"] = mistral_api_key
            # Uplad Mistral Client in the pipeline
            st.session_state.rag_pipeline.mistral_client = Mistral(api_key=mistral_api_key)
        
        #st.header("Carica Documenti")

        #     # --- SEZIONE LOG DI SISTEMA ---
        # with st.expander("Log di Sistema", expanded=False):
        #     if st.button("Pulisci Log"):
        #         st.session_state.log_output = io.StringIO()
        #         logging.basicConfig(
        #             level=logging.INFO,
        #             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        #             stream=st.session_state.log_output
        #         )
        #         logger.info("Log puliti")

        # Visualizza i log
        #st.text_area("Log", st.session_state.log_output.getvalue(), height=300)
        uploaded_files = st.file_uploader(
            "Load a pdf document",
            type="pdf",
            accept_multiple_files=True,
            key = "uploader" # Key for later cleanup of uploaded files
        )
        
        # Check if files have been uploaded
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # 1) If a file has already been processed in this session, skip it
                if uploaded_file.name in st.session_state.processed_files:
                    continue 
                # 2) If a file is already in uploaded_files (already indexed in the past), mark as processed and skip
                if uploaded_file.name in st.session_state.uploaded_files:
                    st.session_state.processed_files.add(uploaded_file.name)
                    continue
                # Check if the file is not already in the session state
                # if uploaded_file.name not in st.session_state.uploaded_files:
                    # Show a spinner during processing
                with st.spinner(f"Document processing {uploaded_file.name}..."):
                    try:
                        # Log the start of processing
                        logger.info(f"Start document processing: {uploaded_file.name}")

                        # Save the uploaded PDF to a temporary file with a unique name generated by the system
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                        temp_file.write(uploaded_file.getvalue())  # Write the content of the uploaded file
                        temp_file_path = temp_file.name            # Get the path of the temporary file
                        temp_file.close()                          # Close the temporary file
                        logger.info(f"File temporarily saved in: {temp_file_path}")

                        # Running OCR on the PDF via Mistral
                        logger.info("Starting OCR with Mistral...")
                        text, txt_path = st.session_state.rag_pipeline.process_pdf_with_mistral_ocr(temp_file_path)

                        # If OCR was successful
                        if text:
                            logger.info(f"OCR successfully completed, text size: {len(text)} characters")

                            # First cleaning up - removes previous temporary files
                            st.session_state.rag_pipeline.cleanup_temp_files()

                            # Run chunking and indexing (including table support)
                            total_chunks, table_count = st.session_state.rag_pipeline.process_and_index_document(
                                text,
                                uploaded_file.name
                            )
                            logger.info(f"Indexing completed: {total_chunks} chunks, {table_count} tables")

                            # Initialize the retrieval/QA chain
                            st.session_state.rag_pipeline.setup_retrieval_chain()

                            # Upload session state with the loaded document
                            st.session_state.uploaded_files[uploaded_file.name] = {
                                "path": temp_file_path,
                                "total_chunks": total_chunks,
                                "table_count": table_count
                            }

                            # Second cleaning up - remove temporary files not needed after processing
                            st.session_state.rag_pipeline.cleanup_temp_files()

                            # Show success message to the user
                            st.success(f"✅ Document {uploaded_file.name} successfully processed and indexed!")
                            st.info(f"Found {table_count} tables in the document")
                        else:
                            # If error during OCR, log and show error message
                            logger.error(f"OCR failed for document {uploaded_file.name}")
                            st.error(f"❌ Error while processing the document {uploaded_file.name}")
                            st.info("Check the logs for more details")
                    except Exception as e:
                        # Full exception handling, detailed logging, and user feedback
                        error_details = traceback.format_exc()
                        logger.error(f"Processing error: {str(e)}\n{error_details}")
                        st.error(f"❌ Error while processing the document {uploaded_file.name}: {str(e)}")
                        st.info("Check the logs for more details")

        # List of uploaded documents
        if st.session_state.uploaded_files:
            st.subheader("Documents loaded:")

            # --- RESET (OPTIONAL) OF THE MASTER FOR THE NEXT RUN  --- 
            #  If some action has requested a reset, we remove the widget keys
            if st.session_state.get("_reset_master_widget", False):
                st.session_state.pop("select_all_sources_widget", None)   # <- key del widget
                st.session_state.pop("_prev_select_all_sources", None)    # <- memoria precedente previous memory
                st.session_state.pop("select_all_sources", None)          # <- internal state
                st.session_state["_reset_master_widget"] = False

            # --- MASTER CHECKBOX ---
            # Internal state (not the widget): default True on first run
            internal_master = st.session_state.get("select_all_sources", True)

            # True widget: use a different key to avoid conflicts
            master = st.checkbox(
                "Select All Documents",
                key="select_all_sources_widget",   # <— Key ONLY for the widget
                value=internal_master
            )

            # Check if the master checkbox ("select all sources") state has changed compared to the previous run. If so, update the internal master state
            prev_master = st.session_state.get("_prev_select_all_sources")
            if prev_master is None or master != prev_master:
                # Update the internal state
                st.session_state["select_all_sources"] = master
                st.session_state["_prev_select_all_sources"] = master

                #  Update all individual checkboxes to match the master state
                for doc_name in st.session_state.uploaded_files.keys():
                    st.session_state[f"select_{doc_name}"] = master

                #  Immediately rerun the app so UI reflects updated checkbox states across all document entries
                st.rerun()

            # --- Checkboxes for each document (use the already set state) --- 
            for doc_name, doc_info in st.session_state.uploaded_files.items():
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    st.checkbox(
                        "",
                        key=f"select_{doc_name}",
                        value=st.session_state.get(f"select_{doc_name}", master)
                    )
                with col2:
                    st.write(f"📄 {doc_name} ({doc_info['total_chunks']} chunks, {doc_info['table_count']} tabelle)")

            # --- Unique button to delete selected documents (equivalent to 'Delete All' if master=True) --- 
                # Create the list of selected documents
            if st.button("Delete Selected"):   
                docs_to_remove = [
                    doc for doc in st.session_state.uploaded_files.keys()
                    if st.session_state.get(f"select_{doc}", False)
                ]

                if not docs_to_remove:
                    st.warning("No documents selected. Please select at least one document.")
                else:
                    try:
                        # 1) Remove embeddings from the vectorstore
                        st.session_state.rag_pipeline.remove_documents(docs_to_remove)

                        # 2) Remove temporary files and clean state
                        for doc_name in docs_to_remove:
                            doc_info = st.session_state.uploaded_files.get(doc_name)
                            if doc_info and os.path.exists(doc_info["path"]):
                                try:
                                    os.remove(doc_info["path"])
                                except Exception as e:
                                    logger.warning(f"Unable to remove the file {doc_info['path']}: {e}")

                            # Remove from state the document and its checkbox
                            st.session_state.uploaded_files.pop(doc_name, None)
                            st.session_state.pop(f"select_{doc_name}", None)

                        # If no documents remain, reset pipeline and chat history
                        if not st.session_state.uploaded_files:
                            st.session_state.rag_pipeline = RagPipeline()
                            st.session_state.chat_history = []
                            # Run clean reset of the master on next run
                            st.session_state["_reset_master_widget"] = True

                        st.success("✅ Selected documents removed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error during removal: {str(e)}")

    # Mean Area: chat and answers
    st.header("Chat with your documents")

    # Visualize chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if "sources" in message:
                        with st.expander("Show sources"):
                            for i, source in enumerate(message["sources"]):
                                # Distinguish between normal sources and tables
                                is_table = source.get('is_table', False)
                                icon = "📊" if is_table else "📄"

                                st.markdown(f"**{icon} Fonte {i+1}:** {source['source']}")

                                if is_table:
                                    # Show table preview in structured form
                                    try:
                                        # Converte il formato testuale in DataFrame
                                        lines = source['content'].strip().split('\n')
                                        table_data = []
                                        for line in lines:
                                            if '|' in line:
                                                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                                                if cells:
                                                    table_data.append(cells)

                                        if len(table_data) > 1:  # Almost headers + one row
                                            st.table(table_data)
                                        else:
                                            st.code(source['content'])
                                    except:
                                        # If error, show original text
                                        st.code(source['content'])
                                else:
                                    st.markdown(f"```\n{source['content']}\n```")

    # Input per la query (sempre visibile)
    user_query = st.chat_input("Ask a question about your documents...") 

    if user_query:
        # Check if there are documents before processing the query
        if not st.session_state.uploaded_files:
            st.chat_message("assistant").write("⚠️ Please load before a PDF document from the sidebar first.")
        else:    
            # Use selected documents for the search
            selected_docs = {
                doc: st.session_state.get(f"select_{doc}", False)
                for doc in st.session_state.uploaded_files.keys()
            }
            active_docs = [doc for doc, selected in selected_docs.items() if selected]
            
            # If no documents are selected, warn the user
            if not active_docs:
                st.chat_message("assistant").write("⚠️ Select almost one document for the search.")
                return
                
            # Add the query to the history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            
            # Prepara la risposta
            with st.chat_message("assistant"):
                with st.spinner("Processing answer..."):
                    # Send selected documents to the query
                    result = st.session_state.rag_pipeline.query(
                        user_query,
                        selected_docs=active_docs
                    )

                    # Show the answer
                    st.write(result["answer"])

                    # Information about sources
                    sources = []
                    for doc in result["source_documents"]:
                        #  Check if it is a table source
                        is_table = "table_id" in doc.metadata if doc.metadata else False

                        sources.append({
                            "source": doc.metadata.get("source", "Unknown document"),
                            "content": doc.page_content,
                            "is_table": is_table,
                            "table_id": doc.metadata.get("table_id", "") if is_table else ""
                        })

                    # Show the source documents
                    if sources:
                        with st.expander("Show sources"):
                            for i, source in enumerate(sources):
                                # Distinguish between normal sources and tables
                                icon = "📊" if source['is_table'] else "📄"

                                st.markdown(f"**{icon} Source {i+1}:** {source['source']}")

                                if source['is_table']:
                                    # For tables, try to display them in a structured way
                                    try:
                                        table_content = source['content']
                                        # Extract the tabular part of the text
                                        table_lines = []
                                        capture = False
                                        for line in table_content.split('\n'):
                                            if "Content of the table:" in line:
                                                capture = True
                                                continue
                                            if capture:
                                                table_lines.append(line)

                                        if table_lines:
                                            st.text("Table preview:") 
                                            st.code('\n'.join(table_lines))
                                    except:
                                        st.code(source['content'])
                                else:
                                    st.markdown(f"```\n{source['content']}\n```")

                    # Add the answer to the history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": sources
                    })
    else:
        st.info("👈 Load a PDF document from left sidebar to start.")

# --- FUNCTION MAIN  --- 
def main():
    """Main function to run the application."""
    # Streamlit
    create_streamlit_app()

if __name__ == "__main__":
    main()