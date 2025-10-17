from typing import Dict, Any
from .cost import Cost
from .page_contents import LoadPageContents, PageContents


class Document:
    contents: list[PageContents]
    cost: Cost

    def __init__(self, contents: list[PageContents], cost: Cost):
        self.contents = contents
        self.cost = cost

    @classmethod
    def from_load_data(cls, load_data: list[LoadPageContents]) -> "Document":
        return cls(
            contents=[PageContents.from_load_data(data) for data in load_data],
            cost=Cost.zero_cost(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "contents": [
                {
                    "contents": content.contents,
                    "page_number": content.page_number
                }
                for content in self.contents
            ],
            "cost": self.cost.to_dict()
        }

    def add_cost(self, additional_cost: Cost) -> None:
        """コストを追加"""
        self.cost = self.cost.add_cost(additional_cost)

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

    @property
    def llm_model_name(self) -> str:
        """使用されたLLMモデル名"""
        return self.cost.llm_model_name
