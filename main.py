from langchain_openai import ChatOpenAI
from exparso import parse_document
import os
from datetime import datetime

llm_model = ChatOpenAI(model="gpt-4o")

# PDFをパース
document = parse_document(
      path="test.pdf",
      model=llm_model,
)

# 結果をテキストとして抽出
extracted_text = ""
for page in document.contents:
    extracted_text += f"=== ページ {page.page_number} ===\n"
    extracted_text += page.contents + "\n\n"

# 出力ディレクトリを作成
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# タイムスタンプ付きのファイル名を生成
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"parsed_document_{timestamp}.txt")

# テキストファイルとして保存
with open(output_file, "w", encoding="utf-8") as f:
    f.write(extracted_text)

print(f"パース結果を {output_file} に保存しました")
print(f"総ページ数: {len(document.contents)}")
print(f"処理コスト: 入力トークン={document.cost.input_token}, 出力トークン={document.cost.output_token}")

# 最初のページの内容をプレビュー表示
if document.contents:
    print(f"\n=== プレビュー（ページ1の最初の200文字）===")
    print(document.contents[0].contents[:200] + "..." if len(document.contents[0].contents) > 200 else document.contents[0].contents)
