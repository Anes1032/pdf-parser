from dataclasses import dataclass
from typing import Dict


@dataclass
class Cost:
    input_token: int
    output_token: int
    llm_model_name: str

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            input_token=self.input_token + other.input_token,
            output_token=self.output_token + other.output_token,
            llm_model_name=other.llm_model_name,
        )

    def add_cost(self, other: "Cost") -> "Cost":
        """他のコストを加算して新しいCostオブジェクトを返す"""
        return Cost(
            input_token=self.input_token + other.input_token,
            output_token=self.output_token + other.output_token,
            llm_model_name=self.llm_model_name,
        )

    def to_dict(self) -> Dict[str, int]:
        """辞書形式に変換"""
        return {
            "input_token": self.input_token,
            "output_token": self.output_token,
            "llm_model_name": self.llm_model_name
        }

    @staticmethod
    def zero_cost() -> "Cost":
        return Cost(input_token=0, output_token=0, llm_model_name="")
