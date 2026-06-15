# ============================================================
# multiAgentFactory.py — 通用多智能体工厂 (终极融合版)
# ============================================================
# 融合了面向对象(OOP)的优雅与 YAML 配置驱动的灵活。
# 既可作为包被 import，也可作为独立服务运行。
# ============================================================

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Annotated, TypedDict, Dict, List, Optional, Any

# ---------- 全局代理关闭 ----------
os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"

try:
    import yaml
    from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
except ImportError as e:
    print(f"[!] 缺少依赖: {e.name}")
    print("    运行: pip install langchain-core langchain-openai langgraph pyyaml")
    sys.exit(1)


# ==========================================================
# 1. 状态定义
# ==========================================================
class FactoryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str


# ==========================================================
# 2. 核心大类 (OOP 封装)
# ==========================================================
class MultiAgentFactory:
    """
    通用多智能体工厂 (核心引擎类)
    将所有的节点逻辑、工具生成和图状态收敛到一个对象中，极度解耦。
    """
    def __init__(self, config: dict):
        self.config = config
        self.agents_cfg = config.get("agents", {})
        self.valid_targets = list(self.agents_cfg.keys()) if self.agents_cfg else ["Agent"]

        # 1. 初始化 LLM
        self.llm = self._init_llm()

        # 2. 动态构建工具并绑定
        self.transfer_tool = self._create_transfer_tool()
        self.llm_with_tools = self.llm.bind_tools([self.transfer_tool])

        # 3. 编译图结构
        self.app = self._build_graph()

    def _init_llm(self) -> ChatOpenAI:
        """从配置初始化 LLM"""
        llm_cfg = self.config.get("llm", {})
        return ChatOpenAI(
            model=llm_cfg.get("model", "gpt-4o"),
            base_url=llm_cfg.get("base_url", "http://127.0.0.1:4000/v1"),
            api_key=llm_cfg.get("api_key", "sk-factory"),
            temperature=llm_cfg.get("temperature", 0),
        )

    def _create_transfer_tool(self):
        """动态生成强约束的 Transfer 工具"""
        targets_str = '", "'.join(self.valid_targets)

        @tool
        def transfer_to(target_agent: str, instructions: str):
            """将控制权移交给其他 agent。"""
            return f"已成功将控制权移交给 {target_agent}。附言: {instructions}"

        transfer_tool = transfer_to
        transfer_tool.__doc__ = (
            f"将控制权移交给其他 agent。\n"
            f"target_agent 必须严格是以下之一: \"{targets_str}\"。\n"
            f"instructions: 告诉接手同事他需要做什么。"
        )
        return transfer_tool

    def _worker_node(self, state: FactoryState):
        """泛化工作节点"""
        current_agent = state.get("active_agent", self.valid_targets[0])
        agent_def = self.agents_cfg.get(current_agent, {})
        sys_prompt = agent_def.get("prompt", "你是一个AI助手。")
        end_hint = agent_def.get("end_hint", "")

        print(f"\n[{current_agent} 正在思考...]")

        system_text = sys_prompt + "\n\n你可以使用 transfer_to 工具将任务转交给其他人。"
        if end_hint:
            system_text += f"\n\n【终止条件】满足以下任一条件时停止移交，直接输出最终结果：\n{end_hint}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | self.llm_with_tools
        response = chain.invoke({"messages": state["messages"]})

        if response.content:
            print(f"[{current_agent}]: {response.content}")

        return {"messages": [response]}

    def _handoff_node(self, state: FactoryState):
        """移交处理节点"""
        last_message = state["messages"][-1]

        if not last_message.tool_calls:
            return {"messages": []}

        tool_call = last_message.tool_calls[0]
        if tool_call["name"] == "transfer_to":
            args = tool_call["args"]
            target_agent = args.get("target_agent")
            instructions = args.get("instructions")

            if target_agent not in self.valid_targets:
                print(f"[!] 警告: 未知的目标 '{target_agent}'，忽略移交")
                return {"messages": []}

            sender = state.get("active_agent", "Unknown")
            print(f"\U0001F504 [系统广播]: {sender} 发起了移交 -> 呼叫 {target_agent}!")
            print(f"   [交接单]: {instructions}")

            tool_msg = ToolMessage(
                content=f"控制权已转移至 {target_agent}。请他立刻处理：{instructions}",
                tool_call_id=tool_call["id"],
            )
            return {"messages": [tool_msg], "active_agent": target_agent}

        return {"messages": []}

    def _router_edge(self, state: FactoryState):
        """路由边"""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def _build_graph(self):
        """构建并编译状态机"""
        workflow = StateGraph(FactoryState)
        workflow.add_node("worker", self._worker_node)
        workflow.add_node("tools", self._handoff_node)

        workflow.add_edge(START, "worker")
        workflow.add_conditional_edges("worker", self._router_edge, {"tools": "tools", END: END})
        workflow.add_edge("tools", "worker")

        return workflow.compile()

    def invoke(self, user_input: str, initial_agent: str = None, thread_id: str = None):
        """
        对外暴露的标准执行接口，支持记忆持久化。
        返回最终回复内容字符串。
        """
        if initial_agent is None:
            initial_agent = self.config.get("initial_agent", self.valid_targets[0])
        if thread_id is None:
            thread_id = str(hash(user_input))  # 每次独立请求默认不同 thread

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "active_agent": initial_agent,
        }

        recursion_limit = self.config.get("recursion_limit", 20)
        config_params = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}

        # 收集所有有内容的回复
        responses = []
        for event in self.app.stream(initial_state, config_params):
            pass

        return "\n".join(responses) if responses else ""


# ==========================================================
# 3. 辅助功能：配置加载与 API 服务
# ==========================================================
def load_config(config_path: Optional[str] = None) -> dict:
    path = config_path or os.environ.get("FACTORY_CONFIG")
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    default_cfg = Path(__file__).resolve().parent / "config" / "agents.yaml"
    if default_cfg.exists():
        with open(str(default_cfg), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    print(f"[!] 警告: 未找到配置文件，请运行 python multiAgentFactory.py --init-config")
    return {"llm": {}, "agents": {"Router": {"prompt": "你是一个路由节点"}}}


def create_api_server(factory: MultiAgentFactory):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/run":
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                user_input = data.get("input", "")

                if not user_input:
                    self._send(400, {"error": "Missing 'input' field"})
                    return

                # 调用 OOP 的 invoke，使用 thread_id 隔离每次请求
                thread_id = str(hash(user_input))
                result = factory.invoke(user_input, initial_agent=data.get("agent"), thread_id=thread_id)

                self._send(200, {"status": "done", "reply": result})

            elif self.path == "/health":
                self._send(200, {"status": "ok", "agents": factory.valid_targets})
            else:
                self._send(404, {"error": "Not found"})

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "agents": factory.valid_targets})
            else:
                self._send(404, {"error": "Not found"})

        def _send(self, code: int, body: dict):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body, ensure_ascii=False).encode())

        def log_message(self, fmt, *args):
            print(f"[API] {args[0]}" if args else "")

    port = int(os.environ.get("FACTORY_PORT", 8765))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\U0001F680 API 服务启动: http://0.0.0.0:{port}")
    print(f"   POST /run  - 提交需求")
    print(f"   GET  /health - 健康检查")
    server.serve_forever()


# ==========================================================
# 4. 主程序入口与辅助配置
# ==========================================================
def _generate_default_config(path: str):
    """生成默认 YAML 配置文件"""
    sample = """# ============================================================
# agents.yaml — 多智能体工厂配置文件
# ============================================================

# ---------- LLM 后端配置 ----------
llm:
  model: "gpt-4o"                          # 模型名称
  base_url: "http://127.0.0.1:4000/v1"     # API 地址 (LiteLLM / OpenAI 兼容)
  api_key: "sk-your-key"                   # API Key
  temperature: 0                            # 创造性程度 (0=确定性)

# ---------- 员工名册 (Agent 定义) ----------
# 每个 agent 可选字段:
#   prompt      - 系统提示词 (必填)
#   end_hint    - 终止条件，满足时不再移交 (可选)
agents:
  Architect:
    prompt: |
      你是一个产品架构师。负责拆解需求，不写具体代码。
      规划好技术方案后，务必移交给 Coder。
    end_hint: |
      - 需求已拆解为详细的开发工单
      - 包含目标文件路径和逻辑说明

  Coder:
    prompt: |
      你是一个全栈开发。负责根据架构师的方案写代码。
      如果方案不清晰，移交给 Architect 询问。
    end_hint: |
      - 代码已编写并落盘
      - 输出简短的《中文测试验收标准》

  QA:
    prompt: |
      你是一个测试工程师。负责审查代码。
      如果有 Bug 就移交回 Coder。
    end_hint: |
      - 验收通过，任务完成
      - 代码无任何报错且逻辑符合预期

# ---------- 流水线控制 ----------
initial_agent: "Architect"    # 第一个接活的 agent
recursion_limit: 20           # 最大移交轮次 (防无限循环)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(sample)


def main():
    parser = argparse.ArgumentParser(description="通用多智能体工厂 (终极融合版)")
    parser.add_argument("--config", "-c", help="YAML 配置文件路径")
    parser.add_argument("--input", "-i", help="用户输入需求 (默认交互式)")
    parser.add_argument("--serve", "-s", action="store_true", help="启动 HTTP API 服务")
    parser.add_argument("--init-config", action="store_true", help="生成默认配置文件后退出")
    args = parser.parse_args()

    if args.init_config:
        root = Path(__file__).resolve().parent
        cfg_dir = root / "config"
        cfg_dir.mkdir(exist_ok=True)
        cfg_file = cfg_dir / "agents.yaml"
        if not cfg_file.exists():
            _generate_default_config(str(cfg_file))
            print(f"\U0001F4DD 已成功生成默认配置文件: {cfg_file}")
            print("   请编辑该文件配置你的模型参数后，再次运行 python multiAgentFactory.py")
        else:
            print(f"\U0001F4DD 配置文件已存在: {cfg_file}")
        return

    config = load_config(args.config)
    factory = MultiAgentFactory(config)

    if args.serve:
        create_api_server(factory)
    else:
        user_input = args.input or input("请输入需求: ").strip()
        if not user_input:
            user_input = "帮我用 Python 写一个简单的计算器"
        print("\n" + "=" * 50)
        factory.invoke(user_input)
        print("\n" + "=" * 50 + "\n\U0001F389 流水线执行结束")


if __name__ == "__main__":
    main()
