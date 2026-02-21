from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

class LLMClient:
    def __init__(self):
        self.llm = ChatOllama(
            model=config.LLM_MODEL_NAME,
            temperature=0.2, # Low temp for factual answers
            keep_alive="5m"
        )
        
        # System prompt for Thai Law Expert
        template = """
        คุณคือผู้เชี่ยวชาญด้านกฎหมายไทย หน้าที่ของคุณคือการตอบคำถามกฎหมายอย่างถูกต้อง แม่นยำ และอ้างอิงข้อกฎหมายให้ชัดเจน
        
        ใช้ข้อมูลบริบท (Context) ที่ได้รับด้านล่างนี้ในการตอบคำถาม:
        ---
        {context}
        ---
        
        คำถาม: {question}
        
        แนวทางการตอบ:
        1. สรุปประเด็นข้อกฎหมายที่เกี่ยวข้อง
        2. อ้างอิงมาตราหรือชื่อกฎหมายจาก Context (ถ้ามี)
        3. หากพบข้อมูลขัดแย้งกัน ให้ยึดถือข้อมูลที่มีวันที่ใหม่กว่าหรือมาจากแหล่ง 'Recent Law' (IAPP) เป็นเกณฑ์
        4. หากข้อมูลใน Context ไม่เพียงพอ ให้ระบุว่า "ข้อมูลไม่เพียงพอสำหรับตอบคำถามนี้ตามเอกสารที่มี" อย่ากุเรื่องขึ้นเอง
        5. ใช้ภาษาไทยที่สุภาพและเป็นทางการ
        
        คำตอบ:
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def generate_answer(self, question, documents):
        # Format documents into a single string
        context_text = "\n\n".join([f"[เอกสารที่ {i+1}]: {doc.page_content}" for i, doc in enumerate(documents)])
        
        return self.chain.invoke({
            "question": question,
            "context": context_text
        })
