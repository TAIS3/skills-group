# MultiAgentFactory (通用多智能体工厂) 🚀

> **高度解耦、配置驱动、即插即用的多智能体状态机引擎**

`multiAgentFactory.py` 是一个基于 LangGraph 构建的通用多智能体协同模块。它融合了**面向对象 (OOP) 的优雅封装**与 **YAML 动态配置的灵活性**。你可以将它作为普通的 Python 类导入到你的复杂业务线中（如 OpenClaw），也可以通过一条命令将其作为独立的 HTTP API 服务或终端交互程序启动。

## ✨ 核心特性

* **🤖 动态智能体名册**：通过 YAML 文件或字典动态注入角色（Agent）及其系统提示词，无需修改底层代码即可增删员工。
* **🔀 强约束移交路由**：底层自动根据有效角色列表动态生成 `transfer_to` 工具，并在系统层拦截未知目标的乱交接行为，杜绝大模型幻觉。
* **📦 完美隔离的 OOP 封装**：彻底摒弃全局变量。所有的状态机图、工具生成、LLM 绑定全部收拢在 `MultiAgentFactory` 类中，支持多实例并发运行互不干扰。
* **🔌 多模态接入**：内置终端 CLI、基于 HTTP 的 API Server，以及最原生的 Python `.invoke()` 方法。

---

## 🛠️ 环境准备

请确保你的 Python 环境 (推荐 3.10+) 中安装了以下依赖：


```
pip install langchain-core langchain-openai langgraph pyyaml
```

*注意：代码内部已硬编码 `os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"`，自动免疫本地模型网关（如 LiteLLM）导致的 502 Bad Gateway 报错。*

---

## 📖 调用姿势大全

本模块支持三种完全不同的使用场景，请根据你的工程架构选择最合适的一种：

### 姿势一：作为 Python 包导入 (推荐用于复杂项目/子节点集成)

这是最标准、最优雅的调用方式。将 `multiAgentFactory.py` 放在你的项目目录下，像使用普通类一样实例化它。

```python
# main_app.py
from skill import MultiAgentFactory

# 1. 准备配置字典 (你也可以通过 load_config() 从 YAML 加载)
config = {
    "llm": {
        "model": "gpt-4o",
        "base_url": "[http://127.0.0.1:4000/v1](http://127.0.0.1:4000/v1)", # 指向你的 LiteLLM 或兼容网关
        "api_key": "sk-your-key"
    },
    "agents": {
        "Router": {"prompt": "你是需求分析师，负责将开发任务移交给 Coder。"},
        "Coder": {"prompt": "你是程序员，负责写代码。写完后告诉用户任务完成。"}
    },
    "initial_agent": "Router",
    "recursion_limit": 20
}

# 2. 实例化工厂引擎
factory = MultiAgentFactory(config)

# 3. 触发执行 (支持传入 thread_id 以便未来接入记忆持久化)
user_input = "帮我用 Python 写一个冒泡排序"
final_state = factory.invoke(user_input, thread_id="user_123_session")

# 4. 获取结果
messages = final_state["messages"]
print(f"最终活跃角色: {final_state['active_agent']}")
print(f"最终回复内容: {messages[-1].content}")

```

### 姿势二：终端命令行交互 (CLI) (推荐用于快速测试)

如果你只想快速测试你的 Agent 设定是否合理，可以直接在终端运行该脚本。

**1. 初始化默认配置文件**
第一次运行，建议先生成一个模板配置文件：

```bash
python multiAgentFactory.py --init-config

```

*(这将在脚本同级目录的 `config/` 下生成 `agents.yaml`)*

**2. 交互式聊天测试**

```bash
python multiAgentFactory.py

```

程序会读取 `config/agents.yaml`，并提示你输入需求，随后在控制台打印完整的智能体流转过程。

**3. 单次静默执行**

```bash
python multiAgentFactory.py -i "写一个冒泡排序并交给QA测试"

```

### 姿势三：作为独立 HTTP API 服务运行 (推荐用于微服务架构)

如果你想让其他语言的后端（如 Go, Java）或者前端直接调用这个多智能体工厂，可以启动内置的轻量级 API 服务。

**1. 启动服务**

```bash
python multiAgentFactory.py --serve
# 或者指定自定义配置文件启动
# FACTORY_CONFIG=my_agents.yaml python multiAgentFactory.py --serve

```

*(服务默认运行在 `http://0.0.0.0:8765`，可通过 `FACTORY_PORT` 环境变量修改)*

**2. API 调用示例 (cURL)**
提交任务并获取多智能体协同后的最终结果：

```bash
curl -X POST [http://127.0.0.1:8765/run](http://127.0.0.1:8765/run) \\
     -H "Content-Type: application/json" \\
     -d '{"input": "帮我规划一个登录系统的表结构，然后让Coder写出SQL"}'

```

查看当前工厂有哪些可用的 Agent：

```bash
curl [http://127.0.0.1:8765/health](http://127.0.0.1:8765/health)
# 返回: {"status": "ok", "agents": ["Architect", "Coder", "QA"]}

```

---

## ⚙️ 配置文件说明 (`agents.yaml`)

如果你使用 YAML 驱动，配置文件的结构非常清晰直观：

```yaml
# ---------- LLM 后端配置 ----------
llm:
  model: "gpt-4o"                          # 模型名称
  base_url: "[http://127.0.0.1:4000/v1](http://127.0.0.1:4000/v1)"     # API 地址
  api_key: "sk-your-key"                   
  temperature: 0                            

# ---------- 员工名册 (Agent 定义) ----------
agents:
  Architect:
    prompt: |
      你是一个产品架构师。负责拆解需求，不写具体代码。
      规划好技术方案后，务必移交给 Coder。
  Coder:
    prompt: |
      你是一个全栈开发。负责根据架构师的方案写代码。
      代码写完后，不需要移交，直接输出结果。

# ---------- 流水线控制 ----------
initial_agent: "Architect"    # 默认第一个接客的 agent
recursion_limit: 20           # 防止死循环的最大流转轮次

```
