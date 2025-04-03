# 对话生成系统 (Dialogue Generation System)

## 项目概述

这是一个基于OpenAI Responses API的对话生成系统，使用多Agent架构设计：

1. **Agent 1**: 初始对话生成代理 - 接收用户输入的背景、对话模式、目标、语言和难度，生成结构化对话内容
2. **Agent 2**: 对话风格改编代理 - 接收Agent 1输出的结构化对话和角色特质，生成风格化对话

系统采用模块化设计，支持轻松添加新的Agent和功能扩展。

## 系统架构

```markdown
agent_Responses_API/
├── agents/                 # Agent相关模块
│   ├── __init__.py        # 包初始化文件
│   ├── base.py            # DialogueAgent基类
│   ├── dialogue_agents.py # 具体Agent实现
│   └── registry.py        # Agent注册管理
├── utils/                 # 工具类
│   ├── __init__.py        # 包初始化文件
│   └── file_manager.py    # 文件管理器
├── dialogue_app.py        # 主应用入口点
├── app_config.py          # 应用配置管理
├── 08-dialogue-app-v3.py  # 原始应用（已重构）
└── requirements.txt       # 依赖项
```

## 使用方法

### Step1：安装虚拟环境

Conda 或 venv都可以

### Step2：安装依赖库

```bash
pip install -r requirements.txt
```

### Step3：设置API KEY

```bash
export OPENAI_API_KEY=put your openai api key here
```

### Step4：测试代码

## 扩展指南

### 添加新的Agent

1. 在`agents/dialogue_agents.py`中创建新的Agent类，继承自`DialogueAgent`基类
2. 实现必要的方法，特别是`process()`方法
3. 在`agents/registry.py`的`initialize_default_agents()`方法中注册新Agent

示例：

```python
class NewAgent(DialogueAgent):
    def __init__(self, client, model="o3-mini"):
        super().__init__(client, model)
        self.agent_type = "new_agent_type"
        self.description = "新Agent描述"
    
    def process(self, *args, **kwargs):
        # 实现处理逻辑
        pass

# 在registry.py中注册
def initialize_default_agents(self) -> None:
    if not self._initialized:
        self.register("initial_dialogue", InitialDialogueAgent)
        self.register("style_adaptation", StyleAdaptationAgent)
        self.register("new_agent_type", NewAgent)  # 添加新Agent
        self._initialized = True
```

### 修改UI

在`dialogue_app.py`中添加新Agent的输入和显示组件，并相应地更新处理逻辑。

## 运行环境要求

## Reference source

https://platform.openai.com/docs/overview

https://platform.openai.com/docs/api-reference/introduction

https://www.youtube.com/watch?v=QF0v3dXg0Kg

https://www.nvidia.com/gtc/?ncid=ref-inor-907027

https://platform.openai.com/storage
