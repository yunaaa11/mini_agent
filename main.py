from agent import run_agent

def debug_loop():
    print("=== Agent 调试模式 ===")
    print("输入任务开始测试，输入 exit 退出\n")

    while True:
        task = input("你：").strip()
        if task.lower() in ["exit", "quit"]:
            print("退出调试模式")
            break

        print("\n[Agent 开始处理]\n")
        try:
            result = run_agent(task)
            print("\n[Agent 最终输出]")
            print(result)
        except Exception as e:
            print("\n[发生错误]")
            print(e)

        print("\n------------------------\n")

if __name__ == "__main__":
    debug_loop()
