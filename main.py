import uuid

from agent import run_agent
import asyncio
async def main():
    # 改进：不要写死 user_1，让用户启动时输入，或者自动生成
    user_name = input("请输入您的用户名（直接回车将随机分配 ID）: ").strip()
    session_id = user_name if user_name else f"user_{uuid.uuid4().hex[:6]}"
    
    print(f"--- 当前会话 ID: {session_id} ---")
    while True:
        task = input("\n用户: ")
        if task.lower() in ["exit", "quit"]: break
        print("助理: ", end="", flush=True)
        async for chunk in run_agent(task, session_id):
    # 如果 chunk 里包含特定的系统标记，直接跳过
           if "event:" in chunk or "data:" in chunk:
              continue
           print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    asyncio.run(main())