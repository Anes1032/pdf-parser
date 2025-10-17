"""
Enhanced PDF Parser with VLM support
exparsoのパターンに従って実装
"""

import os
from typing import Optional, List, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from ..models.document import Document, PageContents, Cost
from ..utils.text_utils import TextProcessor
from ..utils.image_utils import ImageProcessor


class EnhancedPDFParser:
    """Enhanced PDF Parser with VLM support"""
    
    def __init__(self, chat_model: Optional[BaseChatModel] = None, vision_model: Optional[BaseChatModel] = None):
        """Enhanced PDF Parser の初期化"""
        self.chat_model = chat_model
        self.vision_model = vision_model
        
        self.text_processor = TextProcessor(self.chat_model)
        self.image_processor = ImageProcessor(self.vision_model)
    
    def process_pdf(self, pdf_path: str, output_dir: str = "./output") -> Document:
        """PDFファイルを完全に処理するメイン関数（ページごとに統合処理）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # PDFの基本情報を取得
        pages_text = self.text_processor.extract_text_by_pages(pdf_path)
        total_pages = len(pages_text)
        print(f"PDF has {total_pages} pages")
        
        # 画像とキャプションを抽出
        images, captions = self.image_processor.extract_images_and_captions(pdf_path, output_dir)
        matches = self.image_processor.match_figures_to_captions(images, captions)
        
        # ページごとに統合処理
        all_page_contents = []
        total_cost = Cost.zero_cost()
        
        for page_num, page_text in enumerate(pages_text, 1):
            print(f"Processing page {page_num}/{total_pages}...")
            
            # このページの画像を取得
            page_images = [img for img in images if img["page_number"] == page_num]
            
            # ページの統合処理
            enhanced_content, page_cost = self._process_single_page(
                page_text, page_images, page_num
            )
            
            page_contents = PageContents(
                contents=enhanced_content,
                page_number=page_num,
                images=page_images
            )
            
            all_page_contents.append(page_contents)
            total_cost = total_cost.add_cost(page_cost)
        
        return Document(contents=all_page_contents, cost=total_cost)
    
    def _process_single_page(self, page_text: str, page_images: List[Dict], page_num: int) -> tuple[str, Cost]:
        """単一ページの統合処理"""
        # テキスト処理
        processed_text, text_cost = self.text_processor.process_text_with_llm(page_text)
        
        # 画像説明を適切な場所に挿入
        if page_images:
            print(f"  Generating image descriptions for page {page_num}...")
            enhanced_content = self.image_processor.insert_image_descriptions_in_text(
                processed_text, page_images
            )
        else:
            enhanced_content = processed_text
        
        return enhanced_content, text_cost
    
    def save_results(self, document: Document, output_dir: str):
        """結果をファイルに保存"""
        output_file_path = os.path.join(output_dir, "enhanced_processed_text.txt")
        
        # 全ページの内容を結合
        all_content = []
        for page_content in document.contents:
            all_content.append(f"=== Page {page_content.page_number} ===")
            all_content.append(page_content.contents)
            all_content.append("")
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_content))
        
        print(f"pdf parser completed. Total pages: {len(document.contents)}")
        
        return {
            "text_file": output_file_path,
            "images_dir": os.path.join(output_dir, "images")
        }
