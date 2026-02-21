import os
import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm
import config

import os
import json
import glob
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm
import config

def load_jsonl_files(directory):
    print(f"Scanning for JSONL files in {directory}...")
    files = glob.glob(os.path.join(directory, "**/*.jsonl"), recursive=True)
    documents = []
    
    for f in files:
        # Extract year/month from filename e.g. "2025-01.jsonl" -> "2025-01"
        base_name = os.path.basename(f)
        year_month = os.path.splitext(base_name)[0]
        
        print(f"Processing {base_name}...")
        
        try:
            with open(f, 'r', encoding='utf-8') as file:
                for line in file:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        
                        # 1. Check Success
                        if item.get('success') is not True or 'data' not in item:
                            continue
                            
                        data = item['data']
                        
                        # 2. Extract Basic Info
                        raw_filename = data.get('file_name', 'Unknown')
                        # Clean filename to be title (remove extension)
                        title = os.path.splitext(raw_filename)[0]
                        
                        # 3. Concatenate Content from Pages
                        ocr_results = data.get('ocr_results', [])
                        if not ocr_results: continue
                        
                        full_content = ""
                        for page in ocr_results:
                            page_text = page.get('markdown_output', '')
                            if page_text:
                                full_content += page_text + "\n\n"
                        
                        full_content = full_content.strip()
                        if not full_content: continue
                        
                        # 4. Split by "มาตรา X" (Regex Strategy) - Support Thai digits
                        sections = re.split(r'(มาตรา\s+[0-9๑-๙\d]+)', full_content)
                        
                        if len(sections) > 1:
                            # sections[0] is preamble/title text
                            # odd indices are "มาตรา X" headers
                            # even indices are content
                            
                            for i in range(1, len(sections), 2):
                                sec_header = sections[i]
                                sec_body = sections[i+1] if i+1 < len(sections) else ""
                                
                                # Construct text for embedding
                                chunk_text = f"กฎหมาย: {title}\nที่มา: IAPP {year_month}\n{sec_header} {sec_body.strip()}"
                                
                                metadata = {
                                    "source": "iapp_2025",
                                    "filename": data.get('pdf_file', raw_filename),
                                    "title": data.get('doctitle', title),
                                    "section_header": sec_header,
                                    "publish_date": data.get('publishDate', year_month),
                                    "category": data.get('category', 'New Law')
                                }
                                documents.append(Document(page_content=chunk_text, metadata=metadata))
                        else:
                            # Fallback: Document without explicit sections
                            actual_title = data.get('doctitle', title)
                            actual_date = data.get('publishDate', year_month)
                            chunk_text = f"กฎหมาย: {actual_title}\nที่มา: IAPP {actual_date}\nเนื้อหา: {full_content}"
                            metadata = {
                                "source": "iapp_2025",
                                "filename": data.get('pdf_file', raw_filename),
                                "title": actual_title,
                                "publish_date": actual_date,
                                "category": data.get('category', 'New Law')
                            }
                            documents.append(Document(page_content=chunk_text, metadata=metadata))
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return documents

def ingest_recent_law():
    if not os.path.exists(config.RECENT_LAW_DIR):
        print(f"Error: Recent law directory not found at {config.RECENT_LAW_DIR}")
        return

    docs = load_jsonl_files(config.RECENT_LAW_DIR)
    if not docs:
        print("No documents to ingest matching criteria.")
        return

    print(f"Prepared {len(docs)} documents from Recent Law dataset.")

    print(f"Initializing Embedding Model: {config.EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    print(f"Persisting to ChromaDB collection: {config.COLLECTION_RECENT}")
    vectorstore = Chroma(
        collection_name=config.COLLECTION_RECENT,
        embedding_function=embeddings,
        persist_directory=config.DB_DIR
    )
    
    batch_size = 100
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i+batch_size]
        vectorstore.add_documents(documents=batch)
        
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_recent_law()

