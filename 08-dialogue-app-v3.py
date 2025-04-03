# -*- coding: utf-8 -*- # Ensure UTF-8 encoding for wider character support

import streamlit as st
# Set page config as the very first Streamlit command
st.set_page_config(layout="wide", page_title="AI Dialogue Personalizer")

from openai import OpenAI, OpenAIError
import re
import os
import json
from dotenv import load_dotenv
import datetime
import uuid

# --- Load Environment Variables ---
try:
    load_dotenv()
    print("Attempted to load .env file.") # Add print statement for debugging
except Exception as e:
    print(f"Error loading .env file (this might be ignorable if not using .env): {e}")

# --- Configuration ---
# Attempt to initialize OpenAI client using environment variables.
API_KEY_VALID = False # Assume invalid until proven otherwise
OPENAI_CLIENT_ERROR_MESSAGE = None # Store specific error message

try:
    # The OpenAI client automatically uses OPENAI_API_KEY, HTTP_PROXY, HTTPS_PROXY
    # environment variables if they are set (thanks to load_dotenv or manual export).
    client = OpenAI()
    # Perform a simple test call to ensure the client is functional (optional but recommended)
    # Using a very cheap model and minimal request for validation
    client.models.list() # 移除 limit 参数
    API_KEY_VALID = True
    print("OpenAI client initialized and validated successfully.")
except OpenAIError as e:
    OPENAI_CLIENT_ERROR_MESSAGE = f"OpenAI API Error during initialization/validation: {e}. Please check your API Key, network connection, and proxy settings (if applicable) in your environment or .env file."
    print(OPENAI_CLIENT_ERROR_MESSAGE) # Log error to console
except Exception as e:
    OPENAI_CLIENT_ERROR_MESSAGE = f"An unexpected error occurred during OpenAI client initialization: {e}. Check library installation and environment."
    print(OPENAI_CLIENT_ERROR_MESSAGE) # Log error to console

# Display error in Streamlit UI if client initialization failed
if not API_KEY_VALID and OPENAI_CLIENT_ERROR_MESSAGE:
    st.error(OPENAI_CLIENT_ERROR_MESSAGE)

# --- Agent 系统实现 ---

class DialogueAgent:
    """
    对话生成代理的基类，提供通用方法和属性
    """
    def __init__(self, model="o3-mini"):
        self.model = model
        self.client = client

    def call_llm_api(self, prompt, tools=None):
        """使用 OpenAI Response API 调用 LLM"""
        try:
            if tools:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    tools=tools
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt
                )
            return response.output_text
        except Exception as e:
            st.error(f"API 调用错误: {e}")
            return None

class InitialDialogueAgent(DialogueAgent):
    """
    Agent 1: 初始对话生成代理
    接收对话背景、模式、目标、语言要求、难度和对话轮数，生成结构化对话
    """
    def generate_dialogue(self, context, dialogue_mode, goal, language, difficulty, num_turns):
        """生成初始对话内容"""
        prompt = self._build_generation_prompt(context, dialogue_mode, goal, language, difficulty, num_turns)
        response = self.call_llm_api(prompt)
        
        try:
            # 尝试解析响应为 JSON 格式
            if response and '{' in response and '}' in response:
                # 提取 JSON 部分
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                dialogue_data = json.loads(json_str)
            else:
                # 如果不是 JSON 格式，将原文本作为内容返回
                dialogue_data = {
                    "original_text": response,
                    "key_points": [],
                    "intentions": []
                }
            return dialogue_data
        except json.JSONDecodeError:
            # 如果 JSON 解析失败，返回原始文本
            return {
                "original_text": response,
                "key_points": [],
                "intentions": []
            }

    def _build_generation_prompt(self, context, dialogue_mode, goal, language, difficulty, num_turns):
        """构建用于生成对话的提示"""
        prompt = f"""
        作为一个专业的对话生成 AI，请根据以下要求创建一段对话：

        对话背景: {context}
        对话模式: {dialogue_mode}
        对话目标: {goal}
        语言要求: {language}
        内容难度: {difficulty}
        对话轮数: {num_turns}轮

        请生成一段自然流畅的对话，包含以下内容并以 JSON 格式返回:
        1. 对话原始文本
        2. 情节关键节点
        3. 对话中隐含的意图与目标

        返回格式示例:
        {{
            "original_text": "对话原始文本",
            "key_points": ["关键点1", "关键点2"],
            "intentions": ["意图1", "意图2"]
        }}
        """
        return prompt

class StyleAdaptationAgent(DialogueAgent):
    """
    Agent 2: 对话风格改编代理
    接收 Agent 1 的结构化对话数据和角色特质，生成风格化对话
    """
    def adapt_dialogue(self, dialogue_data, user_traits, ai_traits, language=None):
        """基于特质改编对话风格"""
        prompt = self._build_adaptation_prompt(dialogue_data, user_traits, ai_traits, language)
        response = self.call_llm_api(prompt)
        return response

    def _build_adaptation_prompt(self, dialogue_data, user_traits, ai_traits, language=None):
        """构建用于风格改编的提示"""
        # 提取对话数据的关键元素
        original_text = dialogue_data.get("original_text", "")
        key_points = dialogue_data.get("key_points", [])
        intentions = dialogue_data.get("intentions", [])
        
        # 将列表转换为文本表示
        key_points_text = "\n".join([f"- {point}" for point in key_points])
        intentions_text = "\n".join([f"- {intent}" for intent in intentions])
        
        # 检测输出语言
        if not language:
            # 检测原始对话是否包含中文
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', original_text))
            if has_chinese:
                language = "中文"
            else:
                # 默认使用英文
                language = "英文"
        
        # 根据语言构造提示
        if language == "英文":
            prompt = f"""
            As a professional dialogue stylist AI, your task is to rewrite the original dialogue based on the given character traits while maintaining the same plot points and intentions of the original dialogue. Please keep the output in English.

            ## Original Dialogue Information
            Original dialogue text:
            {original_text}

            Key points:
            {key_points_text}

            Dialogue intentions:
            {intentions_text}

            ## Character Traits
            User character traits: {user_traits}
            AI character traits: {ai_traits}

            Please follow these requirements:
            1. Maintain all key points and intentions from the original dialogue
            2. Adjust the dialogue style, tone, and descriptions according to the character traits
            3. Keep the format of the dialogue with clear speaker distinctions
            4. IMPORTANT: Keep the output in the SAME LANGUAGE as the original dialogue (English)
            5. Only return the rewritten dialogue text without additional explanations
            """
        else:
            prompt = f"""
            作为一个专业的对话风格改编 AI，你的任务是将原始对话根据给定的角色特质进行改编，同时保持原始对话的情节和意图不变。

            ## 原始对话信息
            对话原文：
            {original_text}

            关键节点：
            {key_points_text}

            对话意图：
            {intentions_text}

            ## 角色特质
            用户角色特质: {user_traits}
            AI 角色特质: {ai_traits}

            请按照以下要求进行改编：
            1. 保持原始对话的全部关键节点和意图
            2. 根据用户和 AI 的角色特质调整对话风格、语调和描述方式
            3. 请保持对话的格式，包括清晰的说话人区分
            4. 重要提示：请保持输出语言与原始对话相同（中文）
            5. 请只返回改编后的对话文本，不需要额外的解释
            """
            
        return prompt

# 保存生成的结构化内容函数
def save_dialogue_data(dialogue_data, context, goal):
    """
    保存 Agent 1 生成的结构化数据为 JSON 和 Markdown 格式
    返回(json_path, md_path)元组
    """
    try:
        # 创建存储目录（如果不存在）
        save_dir = "dialogue_data"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 生成文件名基础部分
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_context = re.sub(r'[^\w\s-]', '', context)[:20].strip().replace(' ', '_')
        base_filename = f"{timestamp}_{safe_context}_{unique_id}"
        
        # JSON 文件路径
        json_filename = f"{save_dir}/{base_filename}.json"
        
        # Markdown 文件路径
        md_filename = f"{save_dir}/{base_filename}.md"
        
        # 添加元数据
        dialogue_data_with_meta = dialogue_data.copy()
        dialogue_data_with_meta["metadata"] = {
            "timestamp": timestamp,
            "context": context,
            "goal": goal
        }
        
        # 保存 JSON 文件
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(dialogue_data_with_meta, f, ensure_ascii=False, indent=2)
        
        # 生成并保存 Markdown 文件
        with open(md_filename, 'w', encoding='utf-8') as f:
            # 写入标题和元数据
            f.write(f"# 对话记录: {context[:30]}...\n\n")
            f.write(f"**生成时间**: {timestamp}\n\n")
            f.write(f"**对话背景**: {context}\n\n")
            f.write(f"**对话目标**: {goal}\n\n")
            
            # 写入原始对话内容
            f.write("## 对话内容\n\n")
            f.write("```\n")
            f.write(dialogue_data.get("original_text", ""))
            f.write("\n```\n\n")
            
            # 写入关键节点
            if dialogue_data.get("key_points"):
                f.write("## 关键节点\n\n")
                for point in dialogue_data.get("key_points", []):
                    f.write(f"- {point}\n")
                f.write("\n")
            
            # 写入对话意图
            if dialogue_data.get("intentions"):
                f.write("## 对话意图\n\n")
                for intent in dialogue_data.get("intentions", []):
                    f.write(f"- {intent}\n")
        
        return (json_filename, md_filename)
    except Exception as e:
        print(f"保存对话数据时出错: {e}")
        return (None, None)

# 更新已存在的对话数据文件
def update_dialogue_files(json_path, dialogue_data, context, goal):
    """
    更新已存在的对话数据文件(JSON和MD)
    """
    try:
        if not json_path or not os.path.exists(json_path):
            return save_dialogue_data(dialogue_data, context, goal)
            
        # 获取基础文件名并构造MD文件路径
        base_path = os.path.splitext(json_path)[0]
        md_path = f"{base_path}.md"
        
        # 从原JSON文件中获取元数据
        metadata = {}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
                if "metadata" in original_data:
                    metadata = original_data["metadata"]
        except Exception:
            # 如果原文件读取失败，使用新的元数据
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata = {
                "timestamp": timestamp,
                "context": context,
                "goal": goal
            }
        
        # 更新对话数据并保留元数据
        dialogue_data_with_meta = dialogue_data.copy()
        dialogue_data_with_meta["metadata"] = metadata
        
        # 更新JSON文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dialogue_data_with_meta, f, ensure_ascii=False, indent=2)
        
        # 更新Markdown文件
        timestamp = metadata.get("timestamp", "")
        with open(md_path, 'w', encoding='utf-8') as f:
            # 写入标题和元数据
            f.write(f"# 对话记录: {context[:30]}...\n\n")
            f.write(f"**生成时间**: {timestamp}\n\n")
            f.write(f"**对话背景**: {metadata.get('context', context)}\n\n")
            f.write(f"**对话目标**: {metadata.get('goal', goal)}\n\n")
            
            # 写入原始对话内容
            f.write("## 对话内容\n\n")
            f.write("```\n")
            f.write(dialogue_data.get("original_text", ""))
            f.write("\n```\n\n")
            
            # 写入关键节点
            if dialogue_data.get("key_points"):
                f.write("## 关键节点\n\n")
                for point in dialogue_data.get("key_points", []):
                    f.write(f"- {point}\n")
                f.write("\n")
            
            # 写入对话意图
            if dialogue_data.get("intentions"):
                f.write("## 对话意图\n\n")
                for intent in dialogue_data.get("intentions", []):
                    f.write(f"- {intent}\n")
        
        return (json_path, md_path)
    except Exception as e:
        print(f"更新对话数据文件时出错: {e}")
        return (None, None)

# 保存最终对话内容函数
def save_final_dialogue(dialogue_text, initial_dialogue_data, user_traits, ai_traits):
    """
    保存 Agent 2 生成的最终对话内容为 JSON 和 Markdown 格式
    返回(json_path, md_path)元组
    """
    try:
        # 创建存储目录（如果不存在）
        save_dir = "final_dialogue_data"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 获取基础文件名并构造文件路径
        metadata = {}
        if initial_dialogue_data and "metadata" in initial_dialogue_data:
            metadata = initial_dialogue_data["metadata"]
            
        context = metadata.get("context", "")
        goal = metadata.get("goal", "")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_context = re.sub(r'[^\w\s-]', '', context)[:20].strip().replace(' ', '_')
        base_filename = f"{timestamp}_{safe_context}_final_{unique_id}"
        
        json_filename = f"{save_dir}/{base_filename}.json"
        md_filename = f"{save_dir}/{base_filename}.md"
        
        final_dialogue_data = {
            "final_text": dialogue_text,
            "user_traits": user_traits,
            "ai_traits": ai_traits,
            "original_dialogue": initial_dialogue_data,
            "metadata": {
                "timestamp": timestamp,
                "context": context,
                "goal": goal
            }
        }
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(final_dialogue_data, f, ensure_ascii=False, indent=2)
        
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# 最终对话: {context[:30]}...\n\n")
            f.write(f"**生成时间**: {timestamp}\n\n")
            f.write(f"**对话背景**: {context}\n\n")
            f.write(f"**对话目标**: {goal}\n\n")
            
            f.write("## 角色特质\n\n")
            f.write(f"**用户角色特质**: {user_traits}\n\n")
            f.write(f"**AI 角色特质**: {ai_traits}\n\n")
            
            f.write("## 最终对话内容\n\n")
            f.write("```\n")
            f.write(dialogue_text)
            f.write("\n```\n\n")
            
            if initial_dialogue_data and "original_text" in initial_dialogue_data:
                f.write("## 初始对话内容\n\n")
                f.write("```\n")
                f.write(initial_dialogue_data["original_text"])
                f.write("\n```\n\n")
                
                if "key_points" in initial_dialogue_data and initial_dialogue_data["key_points"]:
                    f.write("### 关键节点\n\n")
                    for point in initial_dialogue_data["key_points"]:
                        f.write(f"- {point}\n")
                    f.write("\n")
                
                if "intentions" in initial_dialogue_data and initial_dialogue_data["intentions"]:
                    f.write("### 对话意图\n\n")
                    for intent in initial_dialogue_data["intentions"]:
                        f.write(f"- {intent}\n")
        
        return (json_filename, md_filename)
    except Exception as e:
        print(f"保存最终对话内容时出错: {e}")
        return (None, None)

# 更新已存在的最终对话内容文件
def update_final_dialogue_files(json_path, dialogue_text, initial_dialogue_data, user_traits, ai_traits):
    """
    更新已存在的最终对话内容文件(JSON和MD)
    """
    try:
        if not json_path or not os.path.exists(json_path):
            return save_final_dialogue(dialogue_text, initial_dialogue_data, user_traits, ai_traits)
            
        base_path = os.path.splitext(json_path)[0]
        md_path = f"{base_path}.md"
        
        final_dialogue_data = {
            "final_text": dialogue_text,
            "user_traits": user_traits,
            "ai_traits": ai_traits,
            "original_dialogue": initial_dialogue_data,
            "metadata": {
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "context": "",
                "goal": ""
            }
        }
        
        if initial_dialogue_data and "metadata" in initial_dialogue_data:
            final_dialogue_data["metadata"] = initial_dialogue_data["metadata"]
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_dialogue_data, f, ensure_ascii=False, indent=2)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# 最终对话: {final_dialogue_data['metadata'].get('context', '')[:30]}...\n\n")
            f.write(f"**生成时间**: {final_dialogue_data['metadata'].get('timestamp', '')}\n\n")
            f.write(f"**对话背景**: {final_dialogue_data['metadata'].get('context', '')}\n\n")
            f.write(f"**对话目标**: {final_dialogue_data['metadata'].get('goal', '')}\n\n")
            
            f.write("## 角色特质\n\n")
            f.write(f"**用户角色特质**: {user_traits}\n\n")
            f.write(f"**AI 角色特质**: {ai_traits}\n\n")
            
            f.write("## 最终对话内容\n\n")
            f.write("```\n")
            f.write(dialogue_text)
            f.write("\n```\n\n")
            
            if initial_dialogue_data and "original_text" in initial_dialogue_data:
                f.write("## 初始对话内容\n\n")
                f.write("```\n")
                f.write(initial_dialogue_data["original_text"])
                f.write("\n```\n\n")
                
                if "key_points" in initial_dialogue_data and initial_dialogue_data["key_points"]:
                    f.write("### 关键节点\n\n")
                    for point in initial_dialogue_data["key_points"]:
                        f.write(f"- {point}\n")
                    f.write("\n")
                
                if "intentions" in initial_dialogue_data and initial_dialogue_data["intentions"]:
                    f.write("### 对话意图\n\n")
                    for intent in initial_dialogue_data["intentions"]:
                        f.write(f"- {intent}\n")
        
        return (json_path, md_path)
    except Exception as e:
        print(f"更新最终对话内容文件时出错: {e}")
        return (None, None)

# --- Streamlit UI 实现 ---

def main():
    st.title("Carl的课程内容创作Agents👫🏻")
    
    # 侧边栏用于设置
    with st.sidebar:
        st.header("模型设置")
        model = st.selectbox(
            "选择 LLM 模型",
            ["o3-mini", "gpt-4o-mini" "gpt-4o"]
        )
        
        # 增加模式选择
        st.header("创作模式")
        work_mode = st.radio(
            "选择创作模式",
            ["人机协作", "自动模式"],
            help="自动模式：Agent1生成内容自动传给Agent2；人机协作：Agent1生成后，人工编辑再传给Agent2"
        )

    # 创建两列布局
    col1, col2 = st.columns(2)
    
    # Agent 1 输入 (左侧)
    with col1:
        st.header("Agent 1: 初始对话生成")
        
        context = st.text_area(
            "对话背景",
            placeholder="例如：咖啡馆邂逅、办公室会议等",
            height=100
        )
        
        dialogue_mode = st.radio(
            "对话模式",
            options=["AI先说", "用户先说"]
        )
        
        goal = st.text_area(
            "对话目标",
            placeholder="例如：从讨论书籍/兴趣到获取联系方式/邀请读书会",
            height=100
        )
        
        language = st.selectbox(
            "语言要求",
            options=["英文", "中文", "日文", "韩文", "法文", "德文", "西班牙文"]
        )
        
        difficulty = st.select_slider(
            "内容难度",
            options=["A1", "A2", "B1", "B2", "C1", "C2"],
            value="B1",
            help="CEFR语言等级: A1(入门)到C2(精通)"
        )
        
        # 添加对话轮数选择
        num_turns = st.slider(
            "对话轮数",
            min_value=1,
            max_value=20,
            value=5,
            help="设置对话的来回轮数，1轮=用户和AI各说一次"
        )
    
    # Agent 2 输入 (右侧)
    with col2:
        st.header("Agent 2: 对话风格改编")
        
        user_traits = st.text_area(
            "用户角色特质",
            placeholder="例如：30岁男性，喜欢文学，性格内向，说话谨慎...",
            height=100
        )
        
        ai_traits = st.text_area(
            "AI角色特质",
            placeholder="例如：25岁女性，活泼开朗，喜欢旅行，常用'哇'表达惊讶...",
            height=100
        )

    # 分阶段按钮和状态管理
    if 'dialogue_data' not in st.session_state:
        st.session_state.dialogue_data = None
        st.session_state.dialogue_edited = False
        st.session_state.saved_path = None
        st.session_state.final_dialogue = None
        st.session_state.final_dialogue_edited = False
        st.session_state.final_saved_path = None
    
    # 生成初始对话按钮
    col_buttons = st.columns(2)
    
    with col_buttons[0]:
        if st.button("生成初始对话", type="primary"):
            if not context or not goal:
                st.error("请至少填写对话背景和对话目标")
            else:
                with st.spinner("正在生成初始对话..."):
                    # 实例化 Agent 1
                    initial_agent = InitialDialogueAgent(model=model)
                    
                    # 调用 Agent 1 生成初始对话
                    dialogue_data = initial_agent.generate_dialogue(
                        context=context,
                        dialogue_mode=dialogue_mode,
                        goal=goal,
                        language=language,
                        difficulty=difficulty,
                        num_turns=num_turns
                    )
                    
                    # 检查生成是否成功
                    if dialogue_data and "original_text" in dialogue_data:
                        st.session_state.dialogue_data = dialogue_data
                        st.session_state.dialogue_edited = False
                        
                        # 同时保存 JSON 和 Markdown 文件
                        saved_paths = save_dialogue_data(dialogue_data, context, goal)
                        if saved_paths and saved_paths[0]:
                            st.session_state.saved_path = saved_paths
                            st.success(f"已将结构化内容保存至: {saved_paths[0]} 和 {saved_paths[1]}")
                            
                            # 自动模式下直接调用Agent 2处理
                            if work_mode == "自动模式" and user_traits and ai_traits:
                                with st.spinner("自动模式：正在生成最终对话..."):
                                    # 实例化 Agent 2
                                    style_agent = StyleAdaptationAgent(model=model)
                                    
                                    # 调用 Agent 2 进行风格改编
                                    adapted_dialogue = style_agent.adapt_dialogue(
                                        dialogue_data=dialogue_data,
                                        user_traits=user_traits,
                                        ai_traits=ai_traits,
                                        language=language
                                    )
                                    
                                    # 保存最终对话内容
                                    st.session_state.final_dialogue = adapted_dialogue
                                    st.session_state.final_dialogue_edited = False
                                    
                                    # 保存 JSON 和 Markdown 文件
                                    final_saved_paths = save_final_dialogue(adapted_dialogue, dialogue_data, user_traits, ai_traits)
                                    if final_saved_paths:
                                        st.session_state.final_saved_path = final_saved_paths
                                        st.success(f"自动模式：已生成并保存最终对话至: {final_saved_paths[0]} 和 {final_saved_paths[1]}")
                            elif work_mode == "自动模式" and (not user_traits or not ai_traits):
                                st.warning("自动模式：需要填写用户角色特质和AI角色特质才能自动生成最终对话")
                    else:
                        st.error("对话生成失败，请重试或调整输入参数")
    
    # 人机协作模式下编辑和生成最终对话
    if st.session_state.dialogue_data is not None:
        # 显示 Agent 1 的输出
        st.subheader("初始对话内容")
        
        # 显示保存成功消息（如果有）
        if st.session_state.saved_path and len(st.session_state.saved_path) == 2:
            json_path, md_path = st.session_state.saved_path
            st.success(f"已将结构化内容保存至:\n- JSON: {json_path}\n- Markdown: {md_path}")
        
        # 准备编辑器内容
        original_text = st.session_state.dialogue_data.get("original_text", "")
        key_points = st.session_state.dialogue_data.get("key_points", [])
        intentions = st.session_state.dialogue_data.get("intentions", [])
        
        # 人机协作模式下提供编辑功能
        if work_mode == "人机协作":
            # 展示原始内容
            with st.expander("查看结构化内容", expanded=True):
                st.write("### 原始文本")
                edited_text = st.text_area("编辑对话内容", value=original_text, height=250, key="edit_text")
                
                st.write("### 关键节点")
                key_points_text = "\n".join(key_points)
                edited_key_points = st.text_area("编辑关键节点 (每行一个)", value=key_points_text, height=150, key="edit_key_points")
                
                st.write("### 对话意图")
                intentions_text = "\n".join(intentions)
                edited_intentions = st.text_area("编辑对话意图 (每行一个)", value=intentions_text, height=150, key="edit_intentions")
                
                # 处理编辑后的内容
                edited_key_points_list = [p.strip() for p in edited_key_points.split("\n") if p.strip()]
                edited_intentions_list = [i.strip() for i in edited_intentions.split("\n") if i.strip()]
                
                # 更新编辑状态
                if edited_text != original_text or edited_key_points_list != key_points or edited_intentions_list != intentions:
                    st.session_state.dialogue_edited = True
                    
                    # 更新编辑后的对话数据
                    edited_dialogue_data = {
                        "original_text": edited_text,
                        "key_points": edited_key_points_list,
                        "intentions": edited_intentions_list
                    }
                    st.session_state.dialogue_data = edited_dialogue_data
                
                # 确认按钮
                if st.button("确认编辑", key="confirm_edit_initial_dialogue"):
                    st.success("已更新对话内容")
                    
                    # 如果已编辑，重新保存
                    if st.session_state.dialogue_edited:
                        saved_paths = update_dialogue_files(st.session_state.saved_path[0], st.session_state.dialogue_data, context, goal)
                        if saved_paths:
                            st.session_state.saved_path = saved_paths
                            st.success(f"已将编辑后的结构化内容保存至: {saved_paths[0]} 和 {saved_paths[1]}")
        else:
            # 自动模式下仅显示结构化内容
            with st.expander("查看初始对话", expanded=True):
                st.write(original_text)
                
                if key_points:
                    st.subheader("关键节点")
                    for point in key_points:
                        st.markdown(f"- {point}")
                
                if intentions:
                    st.subheader("对话意图")
                    for intent in intentions:
                        st.markdown(f"- {intent}")
        
        # 生成最终对话按钮
        with col_buttons[1]:
            if work_mode == "人机协作":
                button_text = "生成优化对话"
            else:
                button_text = "生成最终对话"
                
            if st.button(button_text, type="primary"):
                if not user_traits or not ai_traits:
                    st.error("请填写用户角色特质和AI角色特质")
                else:
                    with st.spinner("正在生成最终对话..."):
                        # 实例化 Agent 2
                        style_agent = StyleAdaptationAgent(model=model)
                        
                        # 调用 Agent 2 进行风格改编
                        adapted_dialogue = style_agent.adapt_dialogue(
                            dialogue_data=st.session_state.dialogue_data,
                            user_traits=user_traits,
                            ai_traits=ai_traits,
                            language=language
                        )
                        
                        # 保存最终对话内容
                        st.session_state.final_dialogue = adapted_dialogue
                        st.session_state.final_dialogue_edited = False
                        
                        # 保存 JSON 和 Markdown 文件
                        final_saved_paths = save_final_dialogue(adapted_dialogue, st.session_state.dialogue_data, user_traits, ai_traits)
                        if final_saved_paths:
                            st.session_state.final_saved_path = final_saved_paths
                            st.success(f"已将最终对话内容保存至: {final_saved_paths[0]} 和 {final_saved_paths[1]}")

        # 显示最终对话内容 - 移到按钮外部确保无论按钮点击与否都能显示
        if 'final_dialogue' in st.session_state and st.session_state.final_dialogue:
            st.subheader("最终对话 (风格化)")
            
            # 编辑和实时更新功能
            with st.expander("编辑最终对话", expanded=True):
                edited_final_dialogue = st.text_area(
                    "编辑最终对话内容", 
                    value=st.session_state.final_dialogue, 
                    height=300,
                    key="edit_final_dialogue"
                )
                
                # 更新编辑状态
                if edited_final_dialogue != st.session_state.final_dialogue:
                    st.session_state.final_dialogue = edited_final_dialogue
                    st.session_state.final_dialogue_edited = True
                
                # 确认按钮
                if st.button("确认编辑", key="confirm_edit_final_dialogue"):
                    if st.session_state.final_dialogue_edited:
                        # 更新最终对话内容文件
                        if st.session_state.final_saved_path:
                            updated_paths = update_final_dialogue_files(
                                st.session_state.final_saved_path[0],
                                st.session_state.final_dialogue,
                                st.session_state.dialogue_data,
                                user_traits,
                                ai_traits
                            )
                            if updated_paths:
                                st.session_state.final_saved_path = updated_paths
                                st.success(f"已将编辑后的最终对话内容保存至: {updated_paths[0]} 和 {updated_paths[1]}")
                        else:
                            # 如果没有保存过，则保存
                            final_saved_paths = save_final_dialogue(
                                st.session_state.final_dialogue,
                                st.session_state.dialogue_data,
                                user_traits,
                                ai_traits
                            )
                            if final_saved_paths:
                                st.session_state.final_saved_path = final_saved_paths
                                st.success(f"已将编辑后的最终对话内容保存至: {final_saved_paths[0]} 和 {final_saved_paths[1]}")
            
            # 显示最终对话内容
            st.write(st.session_state.final_dialogue)
            
            # 复制按钮
            st.markdown("""<div style='margin-top: 10px;'>
            <button onclick='navigator.clipboard.writeText(document.getElementById("final-dialogue").innerText)' style='background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 4px; cursor: pointer;'>复制对话</button>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div id='final-dialogue' style='display: none;'>{st.session_state.final_dialogue}</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
