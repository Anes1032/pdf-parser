"""
Text processing utilities
"""

import fitz  # PyMuPDF
from typing import Tuple, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from ..models.document import Cost


class TextProcessor:
    """テキスト処理ユーティリティ"""
    
    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model
    
    
    def extract_text_by_pages(self, pdf_path: str) -> List[str]:
        """PDFファイルからページごとにテキストを抽出"""
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            pages_text.append(page_text)
        
        doc.close()
        return pages_text
    
    
    def process_text_with_llm(self, text: str) -> Tuple[str, Cost]:
        """LLMでテキストを処理"""
        system_prompt = (
            "あなたは優秀な文書処理専門家です。"
            "入力されるテキストはPDFから本文を抽出したものです。"
            "本文には内容が崩れている箇所などがあるため、適宜修正してください。"
            "ヘッダー、フッターは削除し、可能な限り内容は保持してください。"
            "出力は処理済みの本文のみとし、説明は不要です。"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}")
        ])
        
        chain = prompt | self.chat_model
        response = chain.invoke({"text": text})
        
        # トークン数を取得（簡易版）
        input_tokens = len(text.split()) * 1.3  # 概算
        output_tokens = len(response.content.split()) * 1.3
        
        return response.content, Cost(int(input_tokens), int(output_tokens))
