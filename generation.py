"""生成侧核心判断：把检索返回的 5 块原文变成带出处的回答。

链路位置：解析→切块→embed→落盘→读索引→embed问题→两路召回→RRF圈池→rerank精排→MMR取5
          →【本模块】拼提示词调 chat →（下一轮）最小页面 + 公网链接

定位：本文件是项目证据链的核心代码。提示词怎么约束、无依据怎么拒答、出处怎么展示，
属于会被现场追问的判断逻辑，口径已由作者拍板并写进下面两个函数的说明。

输入：embed_index.search(question, k=5) 的返回值，list[dict]；
      每项含 text（原文）、source（文件名）、start/end（结构块区间）、score（精排分）。
输出：回答正文 + 引用出处列表，供命令行打印和下一轮页面展示。
"""
import re
from llm_client import ChatResult

# 拒答文案只在这一处定义，提示词约束和拒答检测共用，避免改一处漏一处
REFUSAL_TEXT = "未找到匹配内容"

# 标引格式：[n]，n 从 1 起对应材料编号；预编译一次，不必每条回答重编译
CITATION_RE = re.compile(r"\[(\d+)\]")


def build_messages(question: str, results: list[dict]) -> list[dict]:
    """把问题与 5 块检索结果拼成 chat 用的 messages。

    输入：question 用户原问题；results 为 embed_index.search() 的返回值。
    输出：OpenAI SDK 格式的 messages（list[dict]，每项有 role/content）。
    关键变量：results 每项的 text 是原文、source 是文件名、start:end 是结构块区间；
              文件内已有的示范表 / 填写示范 / 常见错误示例块按既定口径仍留在检索层，
              只能用提示词约束挡，不能声称语料已净化。
    实现口径：
    1. system 约束「只依据给定片段回答」、拒答文案、示范表假数据不得当真实案例输出，
       并要求回答时用 [n] 标引；n 与材料编号一致，从 1 起。
       拒答时只吐固定文案、不吐 [n] 编号——编号没有出处可指，留着只会让读者困惑。
    2. user 里每块带显式编号 [1]、[2]…… 以及来源、块区间和正文，块间空行分隔；
       模型据此可明确知道该标几号，不必靠顺序猜。
    删行后果：本函数若被跳过，模型拿不到语料原文，只能凭参数记忆乱答。
    """
    numbered = []
    for index, r in enumerate(results, start=1):
        numbered.append(
            f"[{index}] 来源:{r['source']} block:{r['start']}:{r['end']}\n正文:{r['text']}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "请根据候选块回答问题，回答只能依据给定片段，"
                f"语料没直答时只回复：{REFUSAL_TEXT}，不要标引任何编号，"
                "示范表里的假身份证、假地址、示范人名不得当真实案例输出，"
                "回答时用 [n] 标注依据了哪一块，n 与片段编号对应。"
            ),
        },
        {
            "role": "user",
            "content": "问题：" + question + "\n以下是检索到的片段：\n" + "\n\n".join(numbered),
        },
    ]

    return messages


def format_answer(chat_result: ChatResult, results: list[dict]) -> tuple[str, list[dict]]:
    """把模型输出与出处坐标组装成可展示的回答。

    输入：chat_result 是 LLMClient.chat() 的返回；results 是同一次检索的 5 块。
    输出：(回答正文, 出处列表)；出处列表每项含 n、source 与 block[start:end]。
    关键变量：chat_result.content 是模型原文；results[i]['source'/'start'/'end'] 是坐标；
              出处要能回指 meta.json 里的原文块，才满足收口判据的「带出处」。
    实现口径：
    1. 正文直接取模型输出，标引原样保留，读者能看见哪句挂着哪个号。
    2. 命中拒答文案则正文保留、出处为空列表——拒了答就不该挂依据。
    3. 按标引首次出现的顺序取对应块；越界编号忽略、重复编号只列一次；
       n 字段回填原文里的编号（不是列表下标），坐标能回指原文块。
    删行后果：不组装出处，回答就成了无出处断言，收口判据不成立。
    """
    answer = chat_result.content

    if REFUSAL_TEXT in answer:
        return answer, []

    citations = []
    seen = set()
    for match in CITATION_RE.finditer(answer):
        n = int(match.group(1)) - 1
        if n < 0 or n >= len(results) or n in seen:
            continue
        seen.add(n)
        citations.append({
            "n": f"[{n + 1}]",
            "source": results[n]["source"],
            "block[start:end]": [results[n]["start"], results[n]["end"]],
        })

    return answer, citations
