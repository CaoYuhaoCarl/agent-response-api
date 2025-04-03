# -*- coding: utf-8 -*- # Ensure UTF-8 encoding for wider character support

from openai import OpenAI
import json
import requests
import logging

class DialogueAgent:
    """
    对话生成代理的基类，提供通用方法和属性
    所有特定的Agent应继承此类并实现自己的方法
    """
    def __init__(self, client, model="o3-mini", api_type="openai"):
        self.model = model
        self.client = client
        self.api_type = api_type  # "openai" or "openrouter"
        self.agent_type = "base"  # 用于标识Agent类型
        self.description = "基础对话代理"  # 简要描述
        
    def get_agent_info(self):
        """获取Agent的基本信息"""
        return {
            "type": self.agent_type,
            "description": self.description,
            "model": self.model,
            "api_type": self.api_type
        }
    
    def call_llm_api(self, prompt, tools=None):
        """使用 LLM API 调用模型，支持 OpenAI 和 OpenRouter"""
        try:
            if self.api_type == "openai":
                return self._call_openai_api(prompt, tools)
            elif self.api_type == "openrouter":
                return self._call_openrouter_api(prompt, tools)
            else:
                error_msg = f"不支持的 API 类型: {self.api_type}"
                logging.error(error_msg)
                return None
        except Exception as e:
            error_msg = f"API 调用错误: {e}"
            logging.error(error_msg)
            return None
    
    def _call_openai_api(self, prompt, tools=None):
        """调用 OpenAI API"""
        try:
            if tools:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=tools
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                
            # 从响应中提取内容
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logging.error("OpenAI API 未返回有效内容")
                return None
        except Exception as e:
            logging.error(f"OpenAI API 调用错误: {e}")
            return None
    
    def _call_openrouter_api(self, prompt, tools=None):
        """调用 OpenRouter API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.get('api_key')}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # 添加工具调用支持，如果相关模型支持
        if tools:
            data["tools"] = tools
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            try:
                # 正确处理OpenRouter的响应结构
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    logging.error(f"OpenRouter API 响应异常: {result}")
                    return None
            except Exception as e:
                error_msg = f"OpenRouter API 响应解析错误: {e}, 响应内容: {result}"
                logging.error(error_msg)
                return None
        else:
            error_msg = f"OpenRouter API 错误 ({response.status_code}): {response.text}"
            logging.error(error_msg)
            return None
    
    def process(self, *args, **kwargs):
        """处理输入并生成输出的抽象方法，子类必须实现此方法"""
        raise NotImplementedError("子类必须实现process方法")
