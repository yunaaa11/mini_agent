from flask import Flask,request,jsonify
from agent import run_agent
from agent_with_rag import run_agent_with_rag
# 创建 Flask 应用实例
app=Flask(__name__)
# Agent API 接口
@app.route("/agent", methods=["POST"])
def agent_api():
    # 从请求体中解析 JSON 数据
    data = request.get_json()
     # 获取用户任务字段
    task = data.get("task")
    # 参数校验：如果没有传 task，返回 400 错误
    if not task:
        return jsonify({"error": "Missing task"}), 400
    # 调用 Agent 执行用户任务
    result = run_agent(task)
     # 将 Agent 结果以 JSON 格式返回给客户端
    return jsonify({"result": result})

if __name__=="__main__":
    # 启动 Flask 开发服务器
    # host=0.0.0.0 表示允许外部访问
    # port=5000 为服务端口
    app.run(host="0.0.0.0",port=5000)