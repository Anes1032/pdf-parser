from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from exparso import parse_document
from pdf_services.pdf_parser import EnhancedPDFParser
import uuid
from datetime import datetime
from pathlib import Path
import aiofiles
import zipfile
import io
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
async def upload_file(
    file: UploadFile = File(...),
    parser_type: str = Form("exparso")
):
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
        
        # 出力ディレクトリを作成
        session_output_dir = OUTPUT_DIR / session_id
        session_output_dir.mkdir(exist_ok=True)
        
        if parser_type == "enhanced":
            # Enhanced PDF Parserを使用
            try:
                chat_model = ChatOpenAI(model="gpt-4o")
                vision_model = ChatOpenAI(model="gpt-4o")
                
                enhanced_parser = EnhancedPDFParser(chat_model=chat_model, vision_model=vision_model)
                document = enhanced_parser.process_pdf(str(temp_file_path), str(session_output_dir))
                file_paths = enhanced_parser.save_results(document, str(session_output_dir))
                
                # セッション情報を保存
                sessions[session_id] = {
                    "parser_type": "enhanced",
                    "output_file": file_paths["text_file"],
                    "images_dir": file_paths["images_dir"],
                    "filename": f"enhanced_parsed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "page_count": document.total_pages,
                    "input_tokens": document.total_input_tokens,
                    "output_tokens": document.total_output_tokens,
                    "created_at": datetime.now(),
                    "has_images": len(document.contents[0].images) > 0 if document.contents and document.contents[0].images else False
                }
                
                preview_text = document.contents[0].contents[:500] + "..." if document.contents and len(document.contents[0].contents) > 500 else document.contents[0].contents if document.contents else ""
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Enhanced Parserでエラーが発生しました: {str(e)}")
            
        else:
            # exparsoパーサーを使用
            try:
                llm_model = ChatOpenAI(model="gpt-4o")
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
                output_filename = f"exparso_parsed_{timestamp}.txt"
                output_file_path = session_output_dir / output_filename
                
                async with aiofiles.open(output_file_path, 'w', encoding='utf-8') as f:
                    await f.write(extracted_text)
                
                # セッション情報を保存
                sessions[session_id] = {
                    "parser_type": "exparso",
                    "output_file": str(output_file_path),
                    "filename": output_filename,
                    "page_count": len(document.contents),
                    "input_tokens": document.cost.input_token,
                    "output_tokens": document.cost.output_token,
                    "created_at": datetime.now(),
                    "has_images": False
                }
                
                preview_text = document.contents[0].contents[:200] + "..." if document.contents and len(document.contents[0].contents) > 200 else document.contents[0].contents if document.contents else ""
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"exparso Parserでエラーが発生しました: {str(e)}")
        
        # 一時ファイルを削除
        temp_file_path.unlink()
        
        return {
            "session_id": session_id,
            "message": f"ファイルの解析が完了しました（{parser_type}パーサー使用）",
            "parser_type": parser_type,
            "page_count": sessions[session_id]["page_count"],
            "input_tokens": sessions[session_id]["input_tokens"],
            "output_tokens": sessions[session_id]["output_tokens"],
            "has_images": sessions[session_id].get("has_images", False),
            "preview": preview_text
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
    
    if session_data["parser_type"] == "enhanced":
        # Enhanced Parserの場合、ZIPファイルでダウンロード
        return await download_enhanced_zip(session_id, session_data)
    else:
        # exparso Parserの場合、テキストファイルでダウンロード
        file_path = Path(session_data["output_file"])
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="ファイルが見つかりません")
        
        return FileResponse(
            path=str(file_path),
            filename=session_data["filename"],
            media_type='text/plain'
        )

async def download_enhanced_zip(session_id: str, session_data: dict):
    """Enhanced Parserの結果をZIPファイルでダウンロード"""
    text_file_path = Path(session_data["output_file"])
    images_dir_path = Path(session_data["images_dir"])
    
    if not text_file_path.exists():
        raise HTTPException(status_code=404, detail="テキストファイルが見つかりません")
    
    # ZIPファイルをメモリ上で作成
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # テキストファイルを追加
        with open(text_file_path, 'r', encoding='utf-8') as f:
            zip_file.writestr(session_data["filename"], f.read())
        
        # 画像ディレクトリが存在する場合、画像ファイルを追加
        if images_dir_path.exists():
            for image_file in images_dir_path.iterdir():
                if image_file.is_file():
                    # images/ディレクトリ内に配置
                    arcname = f"images/{image_file.name}"
                    zip_file.write(image_file, arcname)
    
    zip_buffer.seek(0)
    
    # ZIPファイル名を生成
    zip_filename = f"enhanced_parsed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    # ZIPファイルの内容を取得
    zip_content = zip_buffer.getvalue()
    
    return Response(
        content=zip_content,
        media_type='application/zip',
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
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
