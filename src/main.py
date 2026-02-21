from retriever import ThaiLawRetriever
from llm_client import LLMClient
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("----------------------------------------------------------------")
    print("   Thai Law RAG System (Llama 3.2 Local)   ")
    print("----------------------------------------------------------------")
    
    try:
        retriever = ThaiLawRetriever()
        llm_client = LLMClient()
    except Exception as e:
        print(f"\nError initializing system: {e}")
        print("Please ensure:")
        print("1. Data has been ingested (run src/ingest_core.py etc)")
        print("2. Ollama is running and model llama3.2 is pulled")
        return

    print("\nSystem ready! Type 'exit' to quit.\n")
    
    while True:
        query = input("คำถามกฎหมาย: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
            
        print("\nกำลังค้นหาข้อมูล...")
        try:
            # 1. Retrieve
            docs = retriever.retrieve(query)
            
            if not docs:
                print("ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล")
                continue
                
            # Show retrieved sources (Optional debug)
            # print(f"\nFound {len(docs)} relevant documents.")
            
            # 2. Generate
            print("กำลังเรียบเรียงคำตอบ...")
            answer = llm_client.generate_answer(query, docs)
            
            print("\n" + "="*50)
            print("คำตอบ:")
            print(answer)
            print("="*50 + "\n")
            
            print("Sources used:")
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown')
                print(f"- [{i+1}] {source}") # print basic source info
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
