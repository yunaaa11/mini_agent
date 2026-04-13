from agent import run_agent
import asyncio
async def main():
     task=input("请输入任务:")
     result=await run_agent(task)
     print("Agent输出:")
     print(result)
if __name__ == "__main__":
    asyncio.run(main())