import asyncio
import json
import os
import tempfile
from fastapi import FastAPI, File,HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
from sse_starlette import EventSourceResponse
from agent import run_agent
import uvicorn
from config import Config
from logger import default_logger as logger
from fastapi import Request
import time
import shutil
from rag import add_document_from_text
app=FastAPI()
class TaskRequest(BaseModel):
    task:str
    session_id: str = "default"

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response

@app.get("/")
async def health():
    return {"status":"ok"}

#集成请求日志
@app.post("/agent")
async def agent_api(request:TaskRequest):
    # 验证输入
    if not request.task:
        raise HTTPException(status_code=400, detail="Task is empty")
    # 1. 获取异步生成器
    generator = run_agent(request.task, request.session_id)
    # 2. 定义一个包装器，将 Token 封装进 SSE 的 data 字段中
    async def event_publisher():
        try:
            # 启动时可以先发一个配置包
            yield {"event": "info", "data": json.dumps({"session_id": request.session_id})}
            async for token in generator:
                # 检查连接是否还活着（可选）
                yield {
                    "event": "message",  # 事件类型，前端默认监听 message
                    "data": token,       # 实际传给前端的内容
                }
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {"event": "error", "data": str(e)}
    # 3. 返回标准的 SSE 响应
    return EventSourceResponse(event_publisher())

@app.post("/upload_knowledge")
async def upload_knowledge(file:UploadFile=File(...)):
    """
    上传 .txt 或 .pdf 文件，自动解析并添加到向量知识库（MD5去重）
    """
    filename=file.filename
    logger.info(f"收到文件上传请求:{filename}")
    #校验文件类型
    if not (filename.endswith('.txt') or filename.endswith('.pdf')):
        raise HTTPException(status_code=400,detail="只支持.txt或.pdf文件")
    try:
        #读取文件内容
        content_bytes=await file.read()
        #保存原始文件到knowledge_base文件夹
        if not os.path.exists(Config.KNOWLEDGE_DIR):
            os.makedirs(Config.KNOWLEDGE_DIR)
        save_path=os.path.join(Config.KNOWLEDGE_DIR,filename)
        with open(save_path,"wb") as f:
            f.write(content_bytes)
        logger.info(f"原始文件已经同步保存至:{save_path}")
        
        if filename.endswith('.txt'):
            text=content_bytes.decode('utf-8')
            success=add_document_from_text(text,filename)
            if success:
                return {"message": f"文件 {filename} 已成功添加到知识库"}
            else:
                return {"message": f"文件 {filename} 内容未变化，已跳过"}
        elif filename.endswith('.pdf'):
            #保存临时文件，再用PyPDFLoader解析
            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
                tmp_file.write(content_bytes)
                tmp_path=tmp_file.name
            try:
                    loader=PyPDFLoader(tmp_path)
                    pages=loader.load()
                    #合并所有页文本
                    full_text="\n".join([page.page_content for page in pages])
                    logger.info(f"PDF 解析成功:{filename},共{len(pages)}页")
                    success=add_document_from_text(full_text,filename)
                    if success:
                       return {"message": f"文件 {filename} 已成功添加到知识库"}
                    else:
                        return {"message": f"文件 {filename} 内容未变化，已跳过"}
            finally:
                    #清理临时文件
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"文件上传处理失败；{e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=5000)