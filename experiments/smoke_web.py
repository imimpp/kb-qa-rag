"""公网问答服务零 API 冒烟检查。

目的：在不发任何网络请求、不读取密钥文件内容的前提下，核对 web/app.py 的
顶层语句顺序、路由响应形状与三项限流。做法是把密钥加载与三路模型调用全部
换成桩，再导入服务模块做检查；所有动作都在本地完成。

用法（仓库根下）：
    .venv/Scripts/python.exe experiments/smoke_web.py

边界：本检查只证明代码结构与路由逻辑成立，不证明随包密钥文件真的生效；
真实网络请求未被本脚本拦截或监测，只确认三路调用均由桩承接。
"""
import ast
import os
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
APP_PATH = WEB / "app.py"

# 桩调用计数：每次桩被调用就加一，用来证明链路走的是桩、不是真实模块
CALLS = {
    "load_dotenv": 0,
    "search": 0,
    "embed": 0,
    "rerank": 0,
    "chat": 0,
    "format": 0,
}

FAKE_BLOCKS = [
    {
        "score": 0.9 - i * 0.1,
        "text": f"桩块 {i + 1}",
        "source": "stub.docx",
        "start": i * 10,
        "end": i * 10 + 9,
    }
    for i in range(5)
]
FAKE_ANSWER = "桩回答（零 API，不代表真实回答质量）"
FAKE_CITATIONS = [
    {"n": "[1]", "source": "stub.docx", "block[start:end]": [0, 9]}
]

RESULTS = []


def _record(name, ok, detail):
    """登记一项检查结果：名称、是否通过、证据细节。"""
    RESULTS.append((name, bool(ok), detail))


def _install_stubs():
    """把密钥加载与三路模型调用换成桩，塞进 sys.modules 供服务模块导入。

    服务模块顶层会显式加载密钥文件；这里的加载函数是空操作，
    因此导入过程不接触密钥文件内容。
    """
    dotenv = ModuleType("dotenv")

    def _load_dotenv(*args, **kwargs):
        CALLS["load_dotenv"] += 1
        return None

    dotenv.load_dotenv = _load_dotenv

    embed_index = ModuleType("embed_index")

    def _search(question, k=5):
        CALLS["search"] += 1
        # 真实检索内部会先 embed 再 rerank，这里用桩结果代替并一并计数
        CALLS["embed"] += 1
        CALLS["rerank"] += 1
        return [dict(block) for block in FAKE_BLOCKS[:k]]

    embed_index.search = _search

    llm_client = ModuleType("llm_client")

    class LLMClient:
        """模型客户端桩：三个方法都只计数、不联网。"""

        def embed(self, *args, **kwargs):
            CALLS["embed"] += 1
            return [[0.0] * 1024]

        def rerank(self, *args, **kwargs):
            CALLS["rerank"] += 1
            return []

        def chat(self, messages, model=None):
            CALLS["chat"] += 1
            return messages

    llm_client.LLMClient = LLMClient

    generation = ModuleType("generation")

    def _build_messages(question, results):
        return [{"role": "user", "content": question}]

    def _format_answer(chat_result, results):
        CALLS["format"] += 1
        return FAKE_ANSWER, [dict(item) for item in FAKE_CITATIONS]

    generation.build_messages = _build_messages
    generation.format_answer = _format_answer

    sys.modules["dotenv"] = dotenv
    sys.modules["embed_index"] = embed_index
    sys.modules["llm_client"] = llm_client
    sys.modules["generation"] = generation


def _static_checks():
    """只读源码做静态审，不执行服务模块。"""
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    chdir_at = assert_at = dotenv_import_at = load_at = chain_import_at = None
    for index, stmt in enumerate(tree.body):
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "dotenv":
            if dotenv_import_at is None:
                dotenv_import_at = index
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            if isinstance(stmt, ast.Import):
                names = [item.name for item in stmt.names]
            else:
                names = [stmt.module or ""]
            if any(name in {"embed_index", "llm_client", "generation"} for name in names):
                if chain_import_at is None:
                    chain_import_at = index
        if isinstance(stmt, ast.Assert) and assert_at is None:
            assert_at = index
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "chdir" and chdir_at is None:
                chdir_at = index
            if name == "load_dotenv" and load_at is None:
                load_at = index

    positions = {
        "切目录": chdir_at,
        "钥匙文件断言": assert_at,
        "加载函数导入": dotenv_import_at,
        "显式加载": load_at,
        "链路导入": chain_import_at,
    }
    order_ok = (
        chdir_at is not None
        and assert_at is not None
        and dotenv_import_at is not None
        and load_at is not None
        and chain_import_at is not None
        and chdir_at < assert_at < dotenv_import_at < load_at < chain_import_at
    )
    _record(
        "顶层语句顺序：切目录→钥匙文件断言→加载函数导入→显式加载→链路导入",
        order_ok,
        "位置 " + " ".join(f"{key}={value}" for key, value in positions.items()),
    )

    override_false = any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "load_dotenv"
        and any(
            keyword.arg == "override"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is False
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )
    _record(
        "显式加载不覆盖已设环境变量",
        override_false,
        "override=False" if override_false else "未找到",
    )

    constants = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and node.targets[0].id in {"RATE_LIMIT", "WINDOW_SEC", "MAX_CONCURRENT"}
        ):
            constants[node.targets[0].id] = node.value.value
    constants_ok = (
        constants.get("RATE_LIMIT") == 3
        and constants.get("WINDOW_SEC") == 60
        and constants.get("MAX_CONCURRENT") == 2
    )
    _record(
        "限流三项常量：每分钟三次、窗口六十秒、全局并发二",
        constants_ok,
        repr(constants),
    )

    port_ok = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Attribute)
        and node.args[0].func.attr == "getenv"
        and node.args[0].args
        and getattr(node.args[0].args[0], "value", None) == "PORT"
        and len(node.args[0].args) > 1
        for node in ast.walk(tree)
    )
    _record(
        "端口兜底读取",
        port_ok,
        "int(os.getenv('PORT', 兜底值))" if port_ok else "未找到",
    )

    has_main_guard = any(
        isinstance(stmt, ast.If)
        and isinstance(stmt.test, ast.Compare)
        and getattr(stmt.test.left, "id", None) == "__name__"
        and stmt.test.comparators
        and getattr(stmt.test.comparators[0], "value", None) == "__main__"
        for stmt in tree.body
    )
    _record("入口保护存在", has_main_guard, "有" if has_main_guard else "无")

    banned = ["AbortController", "setTimeout", "signal=", "stream=True", "timeout="]
    hits = [token for token in banned if token in source]
    _record("无流式与超时代码", not hits, f"命中={hits}" if hits else "无")


def _route_checks(app):
    """在桩环境下调用路由，核对响应形状与三项限流。"""
    class _Client:
        host = "127.0.0.1"

    class _Request:
        client = _Client()

    request = _Request()

    before = dict(CALLS)
    answer = app.answer_question("桩问题")
    chain_ok = (
        isinstance(answer, dict)
        and {"answer", "citations"} <= set(answer)
        and answer["answer"] == FAKE_ANSWER
        and isinstance(answer["citations"], list)
        and CALLS["search"] == before["search"] + 1
        and CALLS["chat"] == before["chat"] + 1
        and CALLS["format"] == before["format"] + 1
    )
    _record(
        "四步串联返回带出处的回答且三路均走桩",
        chain_ok,
        f"检索增量={CALLS['search'] - before['search']} "
        f"对话增量={CALLS['chat'] - before['chat']} "
        f"整理解增量={CALLS['format'] - before['format']}",
    )

    empty = app.ask(app.AskRequest(question="   "), request)
    _record(
        "空问题在服务端被拒",
        getattr(empty, "status_code", None) == 400,
        f"状态码={getattr(empty, 'status_code', None)}",
    )

    too_long = app.ask(app.AskRequest(question="问" * 121), request)
    _record(
        "超长问题在服务端被拒",
        getattr(too_long, "status_code", None) == 400,
        f"状态码={getattr(too_long, 'status_code', None)}",
    )

    app._hits.clear()
    codes = []
    for _ in range(4):
        response = app.ask(app.AskRequest(question="桩问题"), request)
        codes.append(getattr(response, "status_code", 200))
    _record(
        "同一来源第四次提问被限流",
        codes[:3] == [200, 200, 200] and codes[3] == 429,
        f"四次状态码={codes}",
    )

    app._hits.clear()
    acquired = [app._slots.acquire(blocking=False) for _ in range(3)]
    slots_ok = acquired == [True, True, False]
    for taken in acquired:
        if taken:
            app._slots.release()
    _record(
        "全局并发上限为二",
        slots_ok,
        f"三次抢占结果={acquired}",
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print("=== 公网问答服务零 API 冒烟检查 ===")
    _static_checks()

    original_cwd = os.getcwd()
    try:
        sys.path.insert(0, str(WEB))
        _install_stubs()
        import app  # 桩已就位：导入过程不读密钥内容、不发网络请求
        _record("服务模块可在桩环境导入", True, "导入成功，三路均为桩")
        _route_checks(app)
    except Exception as exc:
        _record(
            "服务模块导入与路由检查",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        os.chdir(original_cwd)

    print("\n--- 桩调用计数（证明链路走的是桩）---")
    for name, count in CALLS.items():
        print(f"  {name}: {count}")

    print("\n--- 逐项结果 ---")
    failed = 0
    for name, ok, detail in RESULTS:
        mark = "通过" if ok else "未通过"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name}：{detail}")

    print("\n--- 边界与留白 ---")
    print("  密钥文件内容未被本脚本读取；密钥加载函数已换成空操作桩。")
    print("  零 API 跑通只证明代码结构与路由逻辑，不证明随包密钥文件真的生效。")
    print("  真实网络请求未被本脚本拦截或监测；三路调用均由桩承接，真实模块未加载。")
    print(f"\n合计 {len(RESULTS)} 项，未通过 {failed} 项。")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
