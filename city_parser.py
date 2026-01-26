from typing import List, Optional
CITIES=["北京","上海","广州"]
def extract_city(task:str)->str|None:
    #返回策略：
    #只返回第一个出现的城市
    #如果没有城市，返回 None
    for city in CITIES:
        if city in task:
            return city
    return None
#list 保留顺序，set 不保证顺序，而我们要的是：“用户输入里先出现的城市，优先级更高”
def extract_cities(task:str)->List[str]:
    #返回所有命中的城市（按出现顺序去重）
    #多城市识别 + 冲突处理 + 明确策略
    found=[]
    for city in CITIES:
        if city in task and city not in found:
        #用户输入里包含这个城市名且这个城市还没被加进结果列表（去重，多个里面不能重复）
            found.append(city)
    return found