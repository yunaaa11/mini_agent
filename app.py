from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from agent import run_agent
import uvicorn
from logger import default_logger as logger
from fastapi import Request
import time
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

#集成请求日志
@app.post("/agent")
async def agent_api(request:TaskRequest):
    task=request.task
    if not task:
        raise HTTPException(status_code=400,detail="Missing task")
    result=await run_agent(task,request.session_id)
    return {"result":result}
@app.get("/")
async def health():
    return {"status":"ok"}
if __name__=="__main__":
    uvicorn.app(app,host="0.0.0.0",port=5000)