from openai import OpenAI
from geopy.geocoders import Nominatim
from rich import print
import json
import requests

client = OpenAI()

# 1. 定义自定义工具函数：获取指定位置的当前温度
def get_city_info(location):
    geolocator = Nominatim(user_agent="weather_app")
    location_data = geolocator.geocode(location)
    if not location_data:
        raise ValueError(f"无法找到位置：{location}")
    latitude = location_data.latitude
    longitude = location_data.longitude
    try:
        # 使用 current_weather 参数来获取当前天气数据
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise Exception(f"获取天气数据失败: {str(e)}")
    if 'current_weather' not in data:
        raise Exception("当前天气数据不可用")
    # 返回温度数据
    return {
        "temperature": data['current_weather']['temperature']
    }

# 2. 注册工具（描述信息和参数说明）
tools = [{
    "type": "function",
    "name": "get_city_info",
    "description": "获取给定位置的当前温度",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "要获取天气的地点"
            }
        }
    },
    "required": ["location"],
    "additionalProperties": False
}]

# 3. 构造初始对话，提出用户问题
input_messages = [
    {"role": "user", "content": "宁波的天气如何？"}
]

response = client.responses.create(
    model="gpt-4o-mini",
    input=input_messages,
    tools=tools
)

print(f"response: {response.output}")

# 4. 解析模型返回的工具调用信息，并执行工具函数
tool_call = response.output[0]
args = json.loads(tool_call.arguments)
result = get_city_info(args['location'])
print(f"result: {result}")

# 5. 将工具调用信息及其输出结果追加到对话中
input_messages.append(tool_call)
input_messages.append({
    "type": "function_call_output",
    "call_id": tool_call.call_id,  # 修改这里，使用 call_id 键
    "output": json.dumps(result)  # 保持数据结构为 JSON 格式
})

# 6. 再次调用模型，并传入原问题及工具的返回结果
response_2 = client.responses.create(
    model="gpt-4o-mini",
    input=input_messages,
    tools=tools,
)

print(f"response_2: {response_2.output_text}")
