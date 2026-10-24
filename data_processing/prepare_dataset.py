import json
from pathlib import Path

SRC = Path("/home/yzq/paper/llm_data/all_data.json")
DST = Path("/home/yzq/paper/Probe_HalluDynamics_B/data/endo_diag.jsonl")

QUESTION = "请根据以下内镜所见给出内镜诊断。"


def prepare_dataset(input_path=str(SRC), output_path=str(DST)):
    src = Path(input_path)
    dst = Path(output_path)
    data = json.loads(src.read_text(encoding="utf-8"))
    out = []
    for i, row in enumerate(data):
        ctx = row.get("镜下所见", "").strip()
        ans = row.get("内镜诊断", "").strip()
        if not ctx or not ans:
            continue
        sample = {
            "id": row.get("患者唯一id序号", str(i)),
            "question": QUESTION,
            "context": ctx,
            "answer": ans,
        }
        out.append(sample)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("wrote {} samples to {}".format(len(out), dst))
    return dst


def main():
    prepare_dataset()


if __name__ == "__main__":
    main()
