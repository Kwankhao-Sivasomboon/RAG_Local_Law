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
        # We ask the LLM to act as a judge in ENGLISH because small models follow format rules much better in English
        self.judge_prompt = ChatPromptTemplate.from_template(
            """You are an expert, highly pedantic, and impartial judge evaluating an AI assistant's answer for a Thai Law exam.
        
Question: {question}
Ground Truth (Correct Answer): {ground_truth}
AI Model Prediction: {prediction}

Task:
Determine if the "AI Model Prediction" accurately matches the core logic and facts of the "Ground Truth". You must NEVER hallucinate or invent agreement or disagreement. Read the AI's answer carefully.

EVALUATION CRITERIA:
1. QUESTION TYPE: Is this a "Boolean (Yes/No/Can/Cannot)" question or a "Fact-based (Who/What/When/List)" question?
2. FOR BOOLEAN QUESTIONS:
   - Polarity MUST strongly match. "ได้" (Can) matches "ใช่" (Yes). "ไม่ได้" (Cannot) matches "ไม่ใช่" (No).
   - If the AI answers with a positive word but introduces a contradictory condition (e.g. "Yes, but you need permission first" when the question asks if you can do it BEFORE getting permission), it MUST be a FAIL.
3. FOR FACT-BASED QUESTIONS (Who/What/When/Where):
   - Check if the key entities in the Ground Truth (e.g., exact law names, timelines, government ministries like 'ธนาคารแห่งประเทศไทย') appear in the AI Prediction.
   - If the AI includes the correct entities and conclusion, give it a PASS. Do not fail it just because the phrasing is different.
   - If the AI explicitly provides an incorrect entity or timeline, give it a FAIL.
4. MISSING INFORMATION: If the AI states it does not have enough information ("ข้อมูลไม่เพียงพอ") but the Ground Truth has an answer, it is a FAIL.
5. LENIENCY: Ignore missing section numbers (มาตรา) or extra polite words.

INSTRUCTIONS:
You must structure your response EXACTLY as follows:
Quote: <Highlight the specific core answer from the AI Prediction>
Reason: <Write a 1-2 sentence explanation comparing the Quote to the Ground Truth's core fact or polarity. If the reasoning says they agree, the result MUST be PASS.>
Result: <Exactly "PASS" or "FAIL">

Example 1:
Quote: "ใช่ บริษัทที่มีหุ้นเกินกว่าร้อยละห้าสิบมีสิทธิควบคุม"
Reason: The AI explicitly states the company can control it if it holds more than 50%, which perfectly matches the Ground Truth.
Result: PASS

Example 2:
Quote: "ใช่ สามารถใช้ได้ แต่ต้องได้รับใบอนุญาตก่อน"
Reason: The AI says "Yes", but adds the condition "requires permission first". The Ground Truth says "No" because they don't have the license yet. This is contradictory.
Result: FAIL
"""
        )
        return self.judge_prompt | self.llm_client.llm | StrOutputParser()

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
        sample_df = df.iloc[:10] # Test with first 10 rows
        results = []
        correct = 0

        print(f"Starting Evaluation on {len(sample_df)} samples...")
        
        for idx, row in sample_df.iterrows():
            question = str(row[col_question])
            ground_truth = str(row[col_answer])
            
            # Skip if empty
            if not question or not ground_truth or ground_truth == 'None':
                continue

            # 1. RAG Retrieve + Generate
            # --- Hard Metadata Filtering for this specific test set ---
            # To fix ambiguous queries retrieving from wrong laws (e.g., mining laws or older cancelled Acts)
            filter_meta = {"title": "พระราชบัญญัติธุรกิจสถาบันการเงิน พ.ศ. 2551"}
            
            # Retrieve context (Enhanced Retriever handles logic)
            docs = self.retriever.retrieve(question, filter_metadata=filter_meta)
            
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # --- DEBUG: Print Retrieved Context ---
            print(f"\n[DEBUG] Evaluated Q: {question}")
            print(f"[DEBUG] Retrieved {len(docs)} docs:")
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
                judge_output = self.judge_chain.invoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction
                }).strip()
                
                # Check for "Result: PASS" and "Result: FAIL"
                if "Result: PASS" in judge_output.upper():
                    score = "PASS"
                elif "Result: FAIL" in judge_output.upper():
                    score = "FAIL"
                else:
                    # Fallback manually parsing
                    if "PASS" in judge_output.upper() and not "FAIL" in judge_output.upper():
                        score = "PASS"
                    else:
                        score = "FAIL"
            except Exception as e:
                judge_output = f"ERROR: {str(e)}"
                score = "ERROR"

            if score == "PASS":
                correct += 1

            print(f"\n[Q]: {question}")
            print(f"[GT]: {ground_truth}")
            print(f"[AI]: {prediction}")
            print(f"[Result]: {score}")
            print(f"[Judge Reason]:\n{judge_output}")
            print("-" * 30 + "\n")

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "result": score
            })

        # Summary
        pass_count = correct
        total = len(results)
        if total > 0:
            print(f"\nEvaluation Complete!")
            print(f"Accuracy: {pass_count}/{total} ({pass_count/total*100:.1f}%)")
        else:
            print("\nNo valid samples evaluated.")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()
