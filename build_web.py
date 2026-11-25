"""
构建发布包：把检索生成链路的文件拷入 web/ 发布目录。

发布只上传 web/：43 份原始 docx、实验记录、DECISIONS 不在其中；
web/ 需备齐 4 个 .py 副本与 data/index/ 两个索引文件。真源更新后重跑本脚本同步。
"""
from pathlib import Path
import shutil
import filecmp

# 以脚本位置定位仓库根，不依赖启动时的工作目录
ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

# 白名单：只拷这 4 个 python 文件，不拷 run_qa.py，
# 也不拷任何 .env（演示 key 由本人单独填进 web/.env，脚本不碰）
PY_FILES = ["embed_index.py", "generation.py", "llm_client.py", "chunking.py"]

# 索引文件保持和真源一样的相对位置 data/index/，
# 因为 embed_index.py 是按相对路径找这两个文件的
INDEX_FILES = ["vectors.npy", "meta.json"]


def copy_one(src: Path, dst: Path) -> str:
    """拷一个文件并逐字节校验，返回动作说明。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 幂等：内容一致就跳过，不重复写入
    if dst.exists() and filecmp.cmp(src, dst, shallow=False):
        return "已一致，跳过"
    shutil.copy2(src, dst)
    # shallow=False 才是逐字节比对；默认的 shallow=True 只看元信息，防不了内容漂移
    if not filecmp.cmp(src, dst, shallow=False):
        raise RuntimeError(f"拷贝后校验失败：{src} -> {dst}")
    return "已拷贝并校验通过"


def build() -> None:
    for name in PY_FILES:
        src = ROOT / name
        dst = WEB / name
        print(f"{name}: {copy_one(src, dst)}")
    for name in INDEX_FILES:
        src = ROOT / "data" / "index" / name
        dst = WEB / "data" / "index" / name
        print(f"data/index/{name}: {copy_one(src, dst)}")


if __name__ == "__main__":
    build()
    print("构建完成。")
