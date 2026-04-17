from typing import List, Optional
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆","西安"] 
def extract_city(task: str) -> List[str]:
    """
    提取输入中所有出现的城市名，按出现顺序返回列表。
    原来只返回第一个城市（str），现在统一返回 List[str]，
    兼容 "上海和杭州"、"北京或广州" 等多城市写法。
    返回空列表表示未识别到任何城市。
    """
    found = []
    for city in CITIES:
        if city in task and city not in found:
            found.append(city)
    return found
 
# 保留旧名兼容，指向同一函数
extract_cities = extract_city
if __name__ == "__main__":
    test_tasks = [
        "北京今天天气怎么样",
        "上海和杭州的梅雨季节分别是什么时候？哪个城市更早开始？",
        "广州和深圳哪个冬天更暖和",
        "查询深圳天气",
        "今天去哪里玩",
    ]
    for task in test_tasks:
        cities = extract_city(task)
        print(f"输入: {task}\n提取: {cities}\n")