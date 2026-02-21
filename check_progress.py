import sys
import os

# Add src to path to find config.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import config
import chromadb

if os.name == 'nt':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def check_progress():
    try:
        client = chromadb.PersistentClient(path=config.DB_DIR)
        coll = client.get_collection(config.COLLECTION_CORE)
        current_count = coll.count()
        
        # ยอดเต็มประมาณ 458,731
        total_estimate = 458731
        percent = (current_count / total_estimate) * 100
        
        print("-" * 30)
        print(f"Current Documents in DB: {current_count:,}")
        print(f"Estimated Progress: {percent:.2f}%")
        print("-" * 30)
    except Exception as e:
        print(f"Cannot read DB right now (it might be actively writing): {e}")

if __name__ == "__main__":
    check_progress()
