if __name__=="__main__":
    task=input("请输入任务:")
    if "天气" in task:
        from agent import run_agent
        result=run_agent(task)
    else:
        from agent_with_rag import run_agent_with_rag
        from city_parser import extract_city
        city=extract_city(task)
        if not city:
          print("[Error]未识别到城市")
          exit(1)
        result=run_agent_with_rag(task,city)
    print("Agent输出:")
    print(result)
# def debug_loop():
#     print("=== Agent 调试模式 ===")
#     print("输入任务开始测试，输入 exit 退出\n")

#     while True:
#         task = input("你：").strip()
#         if task.lower() in ["exit", "quit"]:
#             print("退出调试模式")
#             break

#         print("\n[Agent 开始处理]\n")
#         try:
#             result = run_agent(task)
#             print("\n[Agent 最终输出]")
#             print(result)
#         except Exception as e:
#             print("\n[发生错误]")
#             print(e)

#         print("\n------------------------\n")

# if __name__ == "__main__":
#     debug_loop()
