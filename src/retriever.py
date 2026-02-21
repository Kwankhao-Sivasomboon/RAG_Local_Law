import config
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import os

# --- Windows Path Fix ---
if os.name == 'nt':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------

class Retriever:
    def __init__(self):
        print("Initializing Professional Hybrid Retriever (Vector + BM25)...")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        
        # 1. Load Vector Stores
        self.core_db = Chroma(
            collection_name=config.COLLECTION_CORE,
            persist_directory=config.DB_DIR,
            embedding_function=self.embeddings
        )
        self.recent_db = Chroma(
            collection_name=config.COLLECTION_RECENT,
            persist_directory=config.DB_DIR,
            embedding_function=self.embeddings
        )

    def _get_bm25_retriever(self, docs):
        """Builds a BM25 retriever from a list of documents."""
        if not docs:
            return None
        return BM25Retriever.from_documents(docs)

    def retrieve(self, query, filter_metadata=None):
        """
        Retrieves documents using Hybrid Search (Semantic + Keyword) with Filtering.
        """
        # 1. Semantic Search (Vector) - เพื่อดูความหมายโดยรวม
        # เราดึงออกมาจำนวนหนึ่งก่อนเพื่อนำมาทำ BM25 เฉพาะกลุ่ม (เพื่อความเร็วและแม่นยำ)
        core_vector_docs = self.core_db.similarity_search(query, k=15, filter=filter_metadata)
        recent_vector_docs = self.recent_db.similarity_search(query, k=15, filter=filter_metadata)
        all_vector_docs = core_vector_docs + recent_vector_docs

        if not all_vector_docs:
            return []

        # 2. Keyword Search (BM25) - เพื่อเน้นเลขมาตรา หรือคำเฉพาะเจาะจง
        # เราสร้าง BM25 จากกลุ่มเอกสารที่ Vector มองว่าน่าจะใช่ (Candidate Pool)
        # วิธีนี้จะเร็วกว่าการทำ BM25 ทั้ง 4 แสนรายการ และช่วย Re-rank ผลลัพธ์ได้ดีมาก
        bm25_retriever = self._get_bm25_retriever(all_vector_docs)
        
        if bm25_retriever:
            bm25_retriever.k = 8
            vector_retriever = self.core_db.as_retriever(search_kwargs={"k": 8, "filter": filter_metadata})
            ensemble = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
            raw_docs = ensemble.invoke(query)[:8]
        else:
            raw_docs = all_vector_docs[:8]

        # 3. DYNAMIC TEXT REPAIR (The "Fast Patch")
        # Replace the stale "มาตรา: ID:..." text with the correctly patched metadata
        fixed_docs = []
        for doc in raw_docs:
            if 'section_id' in doc.metadata:
                new_label = doc.metadata['section_id']
                # Search and replace "มาตรา: ID:XXXXXX" or "มาตรา: ID:Unknown"
                import re
                doc.page_content = re.sub(r'มาตรา:\s*ID:[a-zA-Z0-9_]+', f'มาตรา: {new_label}', doc.page_content)
            fixed_docs.append(doc)
            
        return fixed_docs
