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
TEST_DATA_PATTERN = config.TEST_DATA_PATTERN
SAMPLE_SIZE = config.EVAL_SAMPLE_SIZE

class Evaluator:
    def __init__(self):
        print("Initializing RAG Components...")
        self.retriever = Retriever()
        
        self.llm_client = LLMClient() # Using the same LLM for generation
        # We need a separate judge chain (can use same model but different prompt)
        self.judge_chain = self._create_judge_chain()
        self.sample_size = SAMPLE_SIZE

    def _create_judge_chain(self):
        # We ask the LLM to act as a judge in ENGLISH because small models follow format rules much better in English
        self.judge_prompt = ChatPromptTemplate.from_template(
            """You are a strict and impartial judge scoring an AI's answer.
        
Question: {question}
Ground Truth (The Correct Answer): {ground_truth}
AI Model Prediction: {prediction}

Task:
Analyze the "Ground Truth" and "AI Model Prediction" to determine their semantic polarity and if they match.

POLARITY DEFINITIONS:
- POSITIVE: Permitted, allowed, can do, yes, ได้, ใช่, ทำได้, อนุญาต.
- NEGATIVE: Prohibited, forbidden, cannot do, no, ไม่ได้, ไม่ใช่, ห้าม, ไม่อนุญาต, ไม่สามารถทำได้.
- OTHER: No information, not enough info, ข้อมูลไม่เพียงพอ, or cannot determine.

RULES:
1. Synonyms are a PASS. "ไม่ได้" (Cannot) matches "ไม่สามารถทำได้", "ไม่อนุญาต", or "ไม่ให้อยู่ภายใต้บังคับ". They all mean the same negative polarity.
2. If the AI says "ข้อมูลไม่เพียงพอ" (Not enough info), its polarity is OTHER, and Result is FAIL (unless Ground Truth is also OTHER).
3. If both AI and Ground Truth agree on the final permission (e.g. they both say "No"), it is a PASS, even if the wording is slightly different.

INSTRUCTIONS:
Output EXACTLY 5 lines:
Ground Truth Polarity: <POSITIVE / NEGATIVE / OTHER>
Prediction Polarity: <POSITIVE / NEGATIVE / OTHER>
Result: <PASS / FAIL>
Quote: <the final conclusion from the AI, e.g., สรุป: ไม่ได้>
Reason: <brief explanation of why they agree or disagree and how their polarities match/mismatch>

Example 1:
Ground Truth Polarity: NEGATIVE
Prediction Polarity: NEGATIVE
Result: PASS
Quote: สรุป: ไม่ได้
Reason: Ground Truth is negative ('ไม่สามารถทำได้') and AI Prediction is negative ('ไม่ได้'). They agree.

Example 2:
Ground Truth Polarity: POSITIVE
Prediction Polarity: OTHER
Result: FAIL
Quote: ข้อมูลไม่เพียงพอ
Reason: The Ground Truth says the action is permitted (Positive), but the AI says 'ข้อมูลไม่เพียงพอ' (Other). They disagree.
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
        
        # Sampling
        sample_df = df.iloc[:self.sample_size] if len(df) > self.sample_size else df
        results = []
        correct = 0

        # Initialize confusion metrics
        # Rows: Actual (GT) -> POSITIVE, NEGATIVE, OTHER
        # Columns: Predicted (AI) -> POSITIVE, NEGATIVE, OTHER
        matrix = {
            "POSITIVE": {"POSITIVE": 0, "NEGATIVE": 0, "OTHER": 0},
            "NEGATIVE": {"POSITIVE": 0, "NEGATIVE": 0, "OTHER": 0},
            "OTHER": {"POSITIVE": 0, "NEGATIVE": 0, "OTHER": 0}
        }

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
                
                # Parse Polarity and Results
                gt_polarity = "OTHER"
                pred_polarity = "OTHER"
                score = "FAIL"
                reason = ""
                quote = ""
                
                for line in judge_output.split("\n"):
                    line_clean = line.strip()
                    if line_clean.upper().startswith("GROUND TRUTH POLARITY:"):
                        gt_polarity = line_clean.split(":", 1)[1].strip().upper()
                    elif line_clean.upper().startswith("PREDICTION POLARITY:"):
                        pred_polarity = line_clean.split(":", 1)[1].strip().upper()
                    elif line_clean.upper().startswith("RESULT:"):
                        score = line_clean.split(":", 1)[1].strip().upper()
                    elif line_clean.upper().startswith("QUOTE:"):
                        quote = line_clean.split(":", 1)[1].strip()
                    elif line_clean.upper().startswith("REASON:"):
                        reason = line_clean.split(":", 1)[1].strip()

                if gt_polarity not in ["POSITIVE", "NEGATIVE", "OTHER"]:
                    gt_polarity = "OTHER"
                if pred_polarity not in ["POSITIVE", "NEGATIVE", "OTHER"]:
                    pred_polarity = "OTHER"
                if "PASS" in score:
                    score = "PASS"
                else:
                    score = "FAIL"
                    
            except Exception as e:
                judge_output = f"ERROR: {str(e)}"
                gt_polarity = "OTHER"
                pred_polarity = "OTHER"
                score = "ERROR"
                reason = str(e)
                quote = ""

            # Update Metrics
            if score == "PASS":
                correct += 1
            
            if gt_polarity in matrix and pred_polarity in matrix[gt_polarity]:
                matrix[gt_polarity][pred_polarity] += 1

            print(f"\n[Q]: {question}")
            print(f"[GT]: {ground_truth} (Polarity: {gt_polarity})")
            print(f"[AI]: {prediction} (Polarity: {pred_polarity})")
            print(f"[Result]: {score}")
            print(f"[Judge Reason]:\n{judge_output}")
            print("-" * 30 + "\n")

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "gt_polarity": gt_polarity,
                "pred_polarity": pred_polarity,
                "result": score
            })

        # Summary & Detailed Metrics Report
        total = len(results)
        if total > 0:
            print(f"\n==============================================================")
            print(f"                     EVALUATION COMPLETE")
            print(f"==============================================================")
            
            pass_count = correct
            accuracy = (pass_count / total) * 100
            
            tp = matrix["POSITIVE"]["POSITIVE"]
            fn = matrix["POSITIVE"]["NEGATIVE"]
            mp = matrix["POSITIVE"]["OTHER"]
            
            fp = matrix["NEGATIVE"]["POSITIVE"]
            tn = matrix["NEGATIVE"]["NEGATIVE"]
            mn = matrix["NEGATIVE"]["OTHER"]
            
            other_pos = matrix["OTHER"]["POSITIVE"]
            other_neg = matrix["OTHER"]["NEGATIVE"]
            other_other = matrix["OTHER"]["OTHER"]
            
            actual_pos = tp + fn + mp
            actual_neg = fp + tn + mn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / actual_pos if actual_pos > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            fpr = fp / actual_neg if actual_neg > 0 else 0.0
            fnr = (fn + mp) / actual_pos if actual_pos > 0 else 0.0
            
            print(f"Accuracy: {pass_count}/{total} ({accuracy:.2f}%)")
            print(f"\nConfusion Matrix:")
            print(f"--------------------------------------------------------------")
            print(f"                      Predicted POS   Predicted NEG   Predicted OTHER")
            print(f"Actual POSITIVE       {tp:<15} {fn:<15} {mp:<15}")
            print(f"Actual NEGATIVE       {fp:<15} {tn:<15} {mn:<15}")
            print(f"Actual OTHER          {other_pos:<15} {other_neg:<15} {other_other:<15}")
            print(f"--------------------------------------------------------------")
            
            print(f"\nClass-Specific Metrics (Positive class = POSITIVE):")
            print(f"  - Precision (Accuracy of positive predictions): {precision*100:.2f}%")
            print(f"  - Recall (Sensitivity/True Positive Rate):     {recall*100:.2f}%")
            print(f"  - F1-Score:                                    {f1*100:.2f}%")
            
            print(f"\nError Rates:")
            print(f"  - False Positive Rate (FPR - Legal Risk):       {fpr*100:.2f}%  (Falsely says YES/Allowed)")
            print(f"  - False Negative Rate (FNR):                   {fnr*100:.2f}%  (Falsely says NO/Forbidden or Unknown)")
            print(f"==============================================================\n")
        else:
            print("\nNo valid samples evaluated.")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()
