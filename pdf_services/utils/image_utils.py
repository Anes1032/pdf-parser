"""
Image processing utilities
"""

import os
import re
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from unstructured.partition.pdf import partition_pdf


class ImageProcessor:
    """画像処理ユーティリティ"""
    
    def __init__(self, vision_model: BaseChatModel):
        self.vision_model = vision_model
    
    def extract_images_and_captions(self, pdf_path: str, output_dir: str) -> Tuple[List[Dict], List[Dict]]:
        """画像とキャプションを抽出"""
        
        # 画像出力ディレクトリを作成
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=images_dir
        )
        
        images = []
        captions = []
        
        for el in elements:
            if el.category == "Image":
                images.append({
                    "id": el.id,
                    "page_number": el.metadata.page_number,
                    "coordinates": el.metadata.coordinates.points if el.metadata.coordinates else None,
                    "image_path": el.metadata.image_path if hasattr(el.metadata, 'image_path') else None,
                    "text": el.text,
                    "category": el.category
                })
            elif el.category == "FigureCaption" or el.text.lower().startswith(("figure", "fig")):
                captions.append({
                    "id": el.id,
                    "page_number": el.metadata.page_number,
                    "coordinates": el.metadata.coordinates.points if el.metadata.coordinates else None,
                    "text": el.text,
                    "category": el.category
                })
        
        return images, captions
    
    def find_midpoint_from_corners(self, points):
        """四角形の座標から中心点を計算"""
        if not points:
            return (0, 0)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return center_x, center_y
    
    def calculate_distance(self, center1, center2):
        """二点間の距離を計算"""
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    def match_figures_to_captions(self, images: List[Dict], captions: List[Dict]) -> List[Tuple[str, Optional[str]]]:
        """図とキャプションをマッチング"""
        matches = []
        
        for image in images:
            image_midpoint = self.find_midpoint_from_corners(image["coordinates"])
            closest_caption = None
            closest_distance = float('inf')
            
            for caption in captions:
                if (image["page_number"] == caption["page_number"] and 
                    self.find_midpoint_from_corners(caption["coordinates"])[1] > image_midpoint[1]):
                    
                    caption_midpoint = self.find_midpoint_from_corners(caption["coordinates"])
                    distance = self.calculate_distance(image_midpoint, caption_midpoint)
                    
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_caption = caption
            
            if closest_caption:
                matches.append((image["id"], closest_caption["id"]))
            else:
                matches.append((image["id"], None))
        
        return matches
    
    def generate_image_description(self, image_path: str, caption: str = "", ocr_text: str = "") -> str:
        """VLMを使用して画像の説明を生成（OCRテキストも活用）"""
        if not os.path.exists(image_path):
            return "画像が見つかりません"
        
        try:
            # 画像を読み込み
            image = Image.open(image_path)
            
            # キャプション情報をプロンプトに追加
            caption_context = f"\n\nこの画像のキャプション情報: {caption}" if caption else ""
            
            # OCRテキスト情報をプロンプトに追加
            ocr_context = ""
            if ocr_text and ocr_text.strip():
                ocr_context = f"\n\nこの画像からOCRで抽出されたテキスト情報: {ocr_text.strip()}"
            
            # VLM用のプロンプト（OCRテキストも活用）
            system_prompt = (
                "あなたは画像を説明する専門家です。画像の内容を詳細に日本語で説明してください。"
                "以下の情報を参考にして、より正確で詳細な説明を提供してください："
                "1. 画像の実際の視覚的内容"
                "2. キャプション情報（参考情報として）"
                "3. OCRで抽出されたテキスト情報（画像内の文字や数値など）"
                "説明では、OCRで抽出されたテキスト情報も適切に組み込んで、"
                "画像の内容をより正確に表現してください。"
            ) + caption_context + ocr_context
            
            human_prompt = (
                "この画像の内容を詳細に説明してください。"
                "画像内の文字、数値、ラベル、グラフの値など、"
                "OCRで抽出されたテキスト情報も含めて、"
                "可能な限り具体的で正確な説明を提供してください。"
            )
            
            # 画像をBase64エンコード
            encoded_image = self._encode_image(image)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", [
                    {"type": "text", "text": human_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ])
            ])
            
            # VLMで画像説明を生成
            chain = prompt | self.vision_model
            response = chain.invoke({})
            
            return response.content
            
        except Exception as e:
            print(f"画像説明生成エラー: {e}")
            return f"画像説明の生成に失敗しました: {str(e)}"
    
    def _encode_image(self, image: Image.Image) -> str:
        """画像をBase64エンコード"""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode()
    
    def generate_page_image_descriptions(self, page_images: List[Dict], page_text: str) -> str:
        """指定されたページの画像説明を生成（captionは本文から除外）"""
        if not page_images:
            return ""
        
        descriptions = []
        
        for image in page_images:
            # 画像ファイル名をデフォルトのfigure_nameとして設定
            image_path = image.get("image_path", "")
            if image_path:
                figure_name = os.path.basename(image_path)
            else:
                figure_name = "Unknown Figure"
            
            figure_type = "Figure"
            
            # 画像のOCRテキストから図番号を抽出
            ocr_text = image.get("text", "")
            if ocr_text:
                # Figure判定
                fig_search = re.search(r'^(fig|figure)\s*(\d+)', ocr_text, re.IGNORECASE)
                if fig_search:
                    figure_name = f"Figure {fig_search.group(2)}"
                    figure_type = "Figure"
                else:
                    # Table判定
                    table_search = re.search(r'^(table|tab)\s*(\d+)', ocr_text, re.IGNORECASE)
                    if table_search:
                        figure_name = f"Table {table_search.group(2)}"
                        figure_type = "Table"
            
            # 画像説明を生成（OCRテキストのみ活用、captionは本文に含まれているため除外）
            image_description = ""
            if image.get("image_path"):
                image_description = self.generate_image_description(image["image_path"], "", ocr_text)
            
            # 画像情報をフォーマット（captionは除外）
            img_info = f"[{figure_type}] {figure_name}"
            
            if ocr_text and ocr_text.strip():
                img_info += f"\n[Image Text]: {ocr_text.strip()}"
            
            if image.get("image_path"):
                image_path = image['image_path']
                if 'images/' in image_path:
                    image_path = image_path.split('images/')[-1]
                    image_path = f"images/{image_path}"
                img_info += f"\n[Image Path]: {image_path}"
            
            if image_description:
                img_info += f"\n[Description]: {image_description}"
            
            descriptions.append(img_info)
        
        return "\n\n".join(descriptions)
    
    def insert_image_descriptions_in_text(self, text: str, page_images: List[Dict]) -> str:
        """テキスト内の適切な場所に画像説明を挿入"""
        if not page_images:
            return text
        
        lines = text.split('\n')
        enhanced_lines = []
        
        # 画像情報を準備
        image_descriptions = self.generate_page_image_descriptions(page_images, text)
        
        # テキスト内で図表の参照を検索して挿入
        i = 0
        while i < len(lines):
            line = lines[i]
            enhanced_lines.append(line)
            
            # 図表の参照パターンを検索
            if re.search(r'(figure|fig|table|図|表)\s*\d+', line, re.IGNORECASE):
                # 図表参照の後に画像説明を挿入
                enhanced_lines.append(f"\n{image_descriptions}\n")
            
            i += 1
        
        # 図表参照が見つからない場合は、ページの最後に追加
        if not any(re.search(r'(figure|fig|table|図|表)\s*\d+', line, re.IGNORECASE) for line in lines):
            enhanced_lines.append(f"\n--- Images ---\n{image_descriptions}")
        
        return '\n'.join(enhanced_lines)
    
