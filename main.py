from agent import run_agent
if __name__=="__main__":
    task = input("请输入任务：")
    result=run_agent(task)
    print("Agent 输出：")
    print(result)