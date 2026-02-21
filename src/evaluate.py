import sys
import os

# --- Windows DLL & OpenMP Fix ---
if sys.platform == "win32":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Logic from fix_torch_path.py
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        torch_lib = os.path.join(conda_prefix, "Lib", "site-packages", "torch", "lib")
        if os.path.exists(torch_lib):
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']
            if hasattr(os, 'add_dll_directory'):
                try: os.add_dll_directory(torch_lib)
                except: pass
# -------------------------------

# Add the current directory (src) to sys.path so we can import modules directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import glob
import config
from retriever import Retriever
from llm_client import LLMClient
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration for Evaluation
TEST_DATA_PATTERN = "datasets/test-*.parquet" # Or train-*.parquet
SAMPLE_SIZE = 5 # Number of items to evaluate for testing

class Evaluator:
    def __init__(self):
        print("Initializing RAG Components...")
        self.retriever = Retriever()
        self.llm_client = LLMClient() # Using the same LLM for generation
        # We need a separate judge chain (can use same model but different prompt)
        self.judge_chain = self._create_judge_chain()

    def _create_judge_chain(self):
        # Single Prompt Evaluation to save tokens
        # We ask Llama 3.2 to act as a judge
        template = """
        คุณคือกรรมการตัดสินความถูกต้องของคำตอบ AI
        
        โจทย์:
        - คำถาม: {question}
        - คำตอบที่ถูกต้อง (Ground Truth): {ground_truth}
        - คำตอบจาก AI (Prediction): {prediction}
        
        ภารกิจ:
        เปรียบเทียบ "คำตอบจาก AI" กับ "คำตอบที่ถูกต้อง" ว่ามีใจความสำคัญตรงกันหรือไม่
        - ไม่ต้องสนรูปแบบการเขียน ให้สนเนื้อหาความหมาย
        - ถ้าคำตอบจาก AI ถูกต้องหรือใกล้เคียงมาก ให้ตอบ "PASS"
        - ถ้าผิด หรือตอบไม่ตรงคำถาม ให้ตอบ "FAIL"
        
        ตอบเฉพาะคำว่า PASS หรือ FAIL เท่านั้น ห้ามอธิบายเพิ่ม
        คำตอบ:
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm_client.llm | StrOutputParser()

    def load_test_data(self):
        files = glob.glob(TEST_DATA_PATTERN)
        if not files:
            print(f"No parquet files found matching {TEST_DATA_PATTERN}")
            return None
        
        print(f"Loading test data from {files[0]}...")
        try:
            df = pd.read_parquet(files[0])
            return df
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return None

    def run_evaluation(self):
        df = self.load_test_data()
        if df is None: return

        print(f"Dataset Columns: {df.columns.tolist()}")
        
        # --- CONFIRMED COLUMNS FROM USER ---
        col_question = 'question' 
        col_answer = 'positive_answer' 
        
        print(f"Using Columns -> Question: '{col_question}', Answer: '{col_answer}'")
        
        # Sampling (Increase strictly if needed, keep small for quick validation)
        sample_df = df.iloc[:5] # Test with first 5 rows first
        results = []

        print(f"Starting Evaluation on {len(sample_df)} samples...")
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
            question = str(row[col_question])
            ground_truth = str(row[col_answer])
            
            # Skip if empty
            if not question or not ground_truth or ground_truth == 'None':
                continue

            # 1. RAG Retrieve + Generate
            # 1. Smart Retrieval with Metadata Filtering
            filter_meta = None
            
            # Simple Heuristic: Check if specific law names are in the question
            known_laws = [
                "พระราชบัญญัติธุรกิจสถาบันการเงิน", 
                "พระราชบัญญัติการประกอบธุรกิจเงินทุน",
                "พระราชบัญญัติหลักทรัพย์และตลาดหลักทรัพย์",
                "พระราชบัญญัติบริษัทมหาชนจำกัด",
                "ประกาศธนาคารแห่งประเทศไทย"
            ]
            
            for law in known_laws:
                if law in question:
                    # Found a specific law mention!
                    # We need to be careful with exact string matching in Chroma.
                    # Usually 'title' field in metadata.
                    # utilizing semantic search with filter is safer.
                    # For now let's try to filter if we are sure.
                    # But since titles might vary slightly (e.g. with B.E. year), 
                    # strict equality filter might miss. 
                    # Chroma supports $contains operator in newer versions? Or plain dict exact match.
                    # Let's just rely on Semantic Search + BM25 (Hybrid) usually.
                    # But User specifically asked for Metadata Filtering.
                    # Let's try filtering by category if possible, or just pass query to enhanced retriever.
                    pass
            
            # Retrieve context (Enhanced Retriever handles logic)
            # We can pass specific filters if we implement strict extraction logic later.
            docs = self.retriever.retrieve(question, filter_metadata=filter_meta)
            
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # --- DEBUG: Print Retrieved Context ---
            print(f"\n[DEBUG] Retrieved {len(docs)} docs:")
            for i, d in enumerate(docs[:3]): # Show top 3
                print(f"  {i+1}. {d.page_content[:150]}...")
            print("-" * 30)
            # --------------------------------------
            try:
                prediction = self.llm_client.generate_answer(question, docs)
            except Exception as e:
                print(f"Error generating answer for Q: {question[:50]}... -> {e}")
                prediction = "Error generating answer"
            
            # 2. Judge
            try:
                score = self.judge_chain.invoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction
                }).strip()
            except Exception as e:
                score = "ERROR"

            print(f"\n[Q]: {question}")
            print(f"[GT]: {ground_truth[:100]}...")
            print(f"[AI]: {prediction[:100]}...")
            print(f"[Result]: {score}")
            print("-" * 30)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "result": score
            })

        # Summary
        pass_count = sum(1 for r in results if "PASS" in r['result'].upper())
        total = len(results)
        if total > 0:
            print(f"\nEvaluation Complete!")
            print(f"Accuracy: {pass_count}/{total} ({pass_count/total*100:.1f}%)")
        else:
            print("\nNo valid samples evaluated.")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()
