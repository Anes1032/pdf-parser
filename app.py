from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from exparso import parse_document
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
import aiofiles
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

app = FastAPI(title="PDF Parser", description="PDF文書を解析・パースするWebアプリケーション")

# テンプレートと静的ファイルの設定
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 出力ディレクトリの作成
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 一時ファイル保存ディレクトリ
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# セッション管理用の辞書（実際の本番環境ではRedis等を使用）
sessions = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """メインページを表示"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """PDFファイルをアップロードして解析"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDFファイルのみアップロード可能です")
    
    # セッションIDを生成
    session_id = str(uuid.uuid4())
    
    try:
        # 一時ファイルに保存
        temp_file_path = TEMP_DIR / f"{session_id}_{file.filename}"
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # LLMモデルを初期化
        llm_model = ChatOpenAI(model="gpt-4o")
        
        # 文書を解析
        document = parse_document(
            path=str(temp_file_path),
            model=llm_model,
        )
        
        # 結果をテキストとして抽出
        extracted_text = ""
        for page in document.contents:
            extracted_text += f"=== ページ {page.page_number} ===\n"
            extracted_text += page.contents + "\n\n"
        
        # 結果ファイルを保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"parsed_document_{timestamp}.txt"
        output_file_path = OUTPUT_DIR / output_filename
        
        async with aiofiles.open(output_file_path, 'w', encoding='utf-8') as f:
            await f.write(extracted_text)
        
        # セッション情報を保存
        sessions[session_id] = {
            "output_file": str(output_file_path),
            "filename": output_filename,
            "page_count": len(document.contents),
            "input_tokens": document.cost.input_token,
            "output_tokens": document.cost.output_token,
            "created_at": datetime.now()
        }
        
        # 一時ファイルを削除
        temp_file_path.unlink()
        
        return {
            "session_id": session_id,
            "message": "ファイルの解析が完了しました",
            "page_count": len(document.contents),
            "input_tokens": document.cost.input_token,
            "output_tokens": document.cost.output_token,
            "preview": document.contents[0].contents[:200] + "..." if document.contents and len(document.contents[0].contents) > 200 else document.contents[0].contents if document.contents else ""
        }
        
    except Exception as e:
        # エラーが発生した場合、一時ファイルを削除
        if temp_file_path.exists():
            temp_file_path.unlink()
        raise HTTPException(status_code=500, detail=f"ファイルの処理中にエラーが発生しました: {str(e)}")

@app.get("/download/{session_id}")
async def download_file(session_id: str):
    """解析結果をダウンロード"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    
    session_data = sessions[session_id]
    file_path = Path(session_data["output_file"])
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")
    
    return FileResponse(
        path=str(file_path),
        filename=session_data["filename"],
        media_type='text/plain'
    )

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """セッションの状態を取得"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    
    return sessions[session_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
