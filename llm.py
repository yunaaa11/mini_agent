from openai import OpenAI
client=OpenAI(
    api_key="sk-0d979069715c4df3a390b43110cbb420",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
def call_llm(messages):
    response=client.chat.completions.create(
        model="qwen-turbo",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content