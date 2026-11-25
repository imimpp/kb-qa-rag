# 公网问答页部署参数（发布前核对用）

本文件只记录发布时要传的参数，不触发发布。发布是对外动作，须本人在当轮明确说「发布/上线」才执行；上一轮的同意不跨轮沿用。参数依据：本地技能文档 `SKILL.md:313-333` 与内置发布工具的运行时参数 schema（两处已核，字段一致）。

## 发布参数（deploy）

| 参数 | 值 | 依据 |
| --- | --- | --- |
| `directory` | `E:/Project/kb-qa-rag/web` | 必填，绝对路径；整目录原样上传，43 份原始 docx、实验记录、DECISIONS 不在 `web/` 内，不会被带上去 |
| `language` | `python` | 显式指定，不用 `auto`：目录里同时有静态页和 Python，避免自动判定走偏 |
| `installCmd` | `pip install -r requirements.txt` | 发布根即 `web/`，依赖清单在根上 |
| `startCmd` | `python app.py` | 入口 `web/app.py` 在上传后的根上；它监听 `$PORT`、绑 `0.0.0.0`（`app.py:31,89`） |
| `appName` | 广西医保经办问答 | 应用显示名，必填；用产品正式名，不用目录名或会话标题 |
| `domainPrefix` | `guangxi-medical-qa` | deploy 必填；小写字母、数字、连字符，≤32 字符；发布前按运行时 schema 再核 |

**明确不传的参数：**

- `entryHtml`：Python 项目的入口是 HTTP 根，不是某个 html 文件；`app.py:48-49` 自己托管静态页。
- 任何环境变量字段：运行时参数全集里没有这一项。演示专用 key 随包放在 `web/.env`，由 `app.py:17` 在顶层显式加载（`override=False`）。
- `port`：不指定，让发布工具注入 `PORT`；`app.py:31` 用 `int(os.getenv("PORT", 8000))` 读取兜底。

## 发布同意闸

- 只有本人在**当轮**明确说「发布/上线」时，才带 `userAskedToPublish: true` 调用发布。
- 发布为覆盖、链接不变；下线后重新发布沿用同一链接。
- 发布与提交是两个独立动作，各自的同意分别等确认。

## 下线（unpublish）

- 传 `action: unpublish` 与同一个 `directory`。
- 不传 `domainPrefix`、`appName`、`language`、`installCmd`、`startCmd` 等 deploy 专属字段。
- 下线会使公网链接失效，须本人明确要求才执行。

## 未确认项（发布前再核，不写死结论）

- 反代超时阈值（通常 30–60 s 量级，未确认）。
- 公网链接的有效期与过期行为（公开文档未写）。
- 托管应用数配额用满后的行为（定价页写体验版可托管 10 个，用满后行为未确认）。
