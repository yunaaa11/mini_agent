from agent import run_agent
import asyncio
async def main():
    while True:
        task = input("\n用户: ")
        if task.lower() in ["exit", "quit"]: break
        
        print("助理: ", end="", flush=True)
        async for chunk in run_agent(task, "user_1"):
            # 过滤标记，或者根据标记改变打印颜色
            if chunk == "[THOUGHT_START]":
                print("\n思考中...", end="", flush=True)
            elif chunk == "[THOUGHT_END]":
                print("\n结论：", end="", flush=True)
            else:
                print(chunk, end="", flush=True)
        print() # 换行
# async def main():
#      task=input("请输入任务:")
#      async for event in run_agent(task):
#       if "messages" in event:
#             last_msg = event["messages"][-1]
#             if hasattr(last_msg, "content"):
#                 print(f"助理: {last_msg.content}")
if __name__ == "__main__":
    asyncio.run(main())