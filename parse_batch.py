"""批量解析 data/raw_doc/*.docx，产物落 data/parsed/*.json。"""

import json
from pathlib import Path

from docx import Document

from parse_docs import parse_one

# 路径锚定脚本所在目录，从任何目录运行都不错位
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data/raw_doc"
OUT_DIR = ROOT / "data/parsed"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.docx"))
    failures = []
    for path in files:
        try:
            result = parse_one(Document(str(path)))
        except Exception as exc:
            # 批处理边界：一份失败不拖死整批，收集起来末尾汇总
            failures.append((path.name, repr(exc)))
            continue
        # 产物自描述：source 记来源文件名，产物文件名被改也能溯源
        payload = {
            "source": path.name,
            "blocks": result["blocks"],
            "placeholders": result["placeholders"],
        }
        out_path = OUT_DIR / (path.stem + ".json")
        # ensure_ascii=False：中文原样写出而非 \uXXXX；indent=2：人能读
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"成功 {len(files) - len(failures)} / {len(files)}")
    for name, err in failures:
        print("失败", name, err)


if __name__ == "__main__":
    main()