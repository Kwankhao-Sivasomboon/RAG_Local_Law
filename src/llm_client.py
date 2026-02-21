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
        # System prompt for Thai Law Expert
        template = """
        คุณคือผู้เชี่ยวชาญด้านกฎหมายไทย หน้าที่ของคุณคือตอบคำถามให้ตรงประเด็น สั้นกระชับ และถูกต้องที่สุด
        
        ใช้ข้อมูลบริบท (Context) ที่ได้รับด้านล่างนี้ในการตอบคำถามเท่านั้น:
        ---
        {context}
        ---
        
        คำถาม: {question}
        
        แนวทางการตอบ (สำคัญมากให้ทำตามนี้):
        1. ให้ตอบประเด็นหลัก "ตรงๆ" ก่อนเป็นอันดับแรก (เช่น ได้ / ไม่ได้ / ใช่ / ไม่ใช่ / มีระบุไว้ / ไม่มี)
        2. หากคำถามถามว่า "ทำได้หรือไม่ก่อนได้รับอนุญาต" และในกฎหมายบอกว่า "ต้องได้รับอนุญาตก่อน" คุณต้องตอบฟันธงว่า "ไม่ได้" หรือ "ไม่ใช่" ห้ามตอบว่า "ได้ แต่ต้องขออนุญาตก่อน" อย่างเด็ดขาด (ห้ามตอบกำกวมแบบตรรกะย้อนแย้ง)
        3. จากนั้นให้อธิบายเหตุผลสั้นๆ โดยอ้างอิงชื่อกฎหมายจาก Context
        4. สำคัญมาก: หากใน Context ไม่ได้ระบุเลข "มาตรา" เอาไว้อย่างชัดเจน ห้ามเดาหรือแต่งเลขมาตราขึ้นมาเองเด็ดขาด ให้อ้างอิงแค่ชื่อกฎหมายหรือพระราชบัญญัติก็พอ
        5. อย่าเขียนยาวเยิ่นเย้อเป็นข้อความยาวๆ หรือลิสต์สรุปประเด็นที่ไม่จำเป็น
        6. หากข้อมูลใน Context ไม่ได้สรุปหรือมีเนื้อหาเกี่ยวข้องกับคำถาม ให้ตอบเพียงแค่ "ข้อมูลไม่เพียงพอ"
        
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
