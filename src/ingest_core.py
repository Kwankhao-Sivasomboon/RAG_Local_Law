import json
import os
import re
import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from tqdm import tqdm
from huggingface_hub import snapshot_download

def ingest_core_law():
    print("Starting Ingestion (Filter: Year 2500+ AND is_latest only)...")
    
    dataset_path = os.path.join(config.DATASETS_DIR, "ocs-krisdika_manual")
    
    all_files = []
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".jsonl"):
                    # --- FILTER YEAR (2500+) ---
                    # ดึงปีจากชื่อโฟลเดอร์ เช่น data/2562/...
                    year_match = re.search(r'(\d{4})', root)
                    if year_match:
                        year = int(year_match.group(1))
                        # แปลง ค.ศ. เป็น พ.ศ. โดยประมาณถ้าเจอตัวเลขน้อยกว่า 2400
                        actual_be_year = year if year > 2400 else year + 543
                        if actual_be_year >= 2500:
                            all_files.append(os.path.join(root, f))
    
    documents = []
    print(f"Processing {len(all_files)} filtered files...")
    
    for file_path in tqdm(all_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    
                    # --- FILTER LATEST ---
                    if not data.get('is_latest', False): continue 
                    
                    title = data.get('title', 'Unknown')
                    sections = data.get('sections', [])
                    
                    for sec in sections:
                        content = sec.get('content', '').strip()
                        if not content: continue
                        
                        # Fix Label: Find 'มาตรา X' from content text
                        match = re.search(r'(มาตรา\s*[0-9๑-๙\./]+)', content)
                        section_label = match.group(1).strip() if match else f"ID:{sec.get('sectionId')}"
                        
                        full_text = f"กฎหมาย: {title}\nมาตรา: {section_label}\nเนื้อหา: {content}"
                        
                        metadata = {
                            "source": "ocs-krisdika",
                            "title": title,
                            "section_id": section_label,
                        }
                        documents.append(Document(page_content=full_text, metadata=metadata))
        except Exception:
            continue

    print(f"Total documents to index: {len(documents)}")
    
    if not documents:
        print("No documents found!")
        return

    # 2. Ingest to ChromaDB (Original Way - No Manual Batching)
    print(f"Initializing Embeddings and Vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    
    # Let LangChain & Chroma handle the batching internally like before
    print(f"Adding {len(documents)} documents to ChromaDB. This will take some time...")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=config.COLLECTION_CORE,
        persist_directory=config.DB_DIR
    )
    
    print("--- Core Ingestion Success! ---")

if __name__ == "__main__":
    ingest_core_law()
