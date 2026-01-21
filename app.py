from flask import Flask,request,jsonify
from agent import run_agent
from agent_with_rag import run_agent_with_rag
app=Flask(__name__)
@app.route("/agent", methods=["POST"])
def agent_api():
    data = request.get_json()
    task = data.get("task")
    if not task:
        return jsonify({"error": "Missing task"}), 400
    result = run_agent(task)
    return jsonify({"result": result})

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)