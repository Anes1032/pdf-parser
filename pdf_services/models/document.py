"""
Document models compatible with exparso format
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Cost:
    """コスト情報"""
    input_token: int = 0
    output_token: int = 0
    
    @classmethod
    def zero_cost(cls) -> "Cost":
        return cls(0, 0)
    
    def add_cost(self, other: "Cost") -> "Cost":
        return Cost(
            self.input_token + other.input_token,
            self.output_token + other.output_token
        )
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "input_token": self.input_token,
            "output_token": self.output_token
        }


@dataclass
class PageContents:
    """ページ内容（exparso形式）"""
    contents: str
    page_number: int
    images: Optional[List[Dict[str, Any]]] = None
    
    def to_exparso_format(self) -> Dict[str, Any]:
        """exparsoのPageContents形式に変換"""
        return {
            "contents": self.contents,
            "page_number": self.page_number
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "contents": self.contents,
            "page_number": self.page_number
        }
        if self.images:
            result["images"] = self.images
        return result


@dataclass
class Document:
    """ドキュメント（exparso形式）"""
    contents: List[PageContents]
    cost: Cost
    
    def to_exparso_format(self) -> Dict[str, Any]:
        """exparsoのDocument形式に変換"""
        return {
            "contents": [content.to_exparso_format() for content in self.contents],
            "cost": self.cost.to_dict()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "contents": [content.to_dict() for content in self.contents],
            "cost": self.cost.to_dict()
        }
    
    @property
    def total_pages(self) -> int:
        """総ページ数"""
        return len(self.contents)
    
    @property
    def total_input_tokens(self) -> int:
        """総入力トークン数"""
        return self.cost.input_token
    
    @property
    def total_output_tokens(self) -> int:
        """総出力トークン数"""
        return self.cost.output_token
