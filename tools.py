def get_weather(city:str) -> str:
    print(f"[Tool] get_weather 被调用，参数 city={city}")
    fake_weather={
        "北京":"晴天,5°C",
        "上海":"多云,8°C",
        "广州": "小雨, 15°C"
    }
    return fake_weather.get(city, "暂无该城市天气数据")

#工具可以是数据库、接口、搜索、计算器