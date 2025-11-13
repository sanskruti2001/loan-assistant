import json, glob
from pathlib import Path
OUT = Path("data/kb.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def simple_clean(text):
    # remove extra whitespace
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def main():
    files = glob.glob("data/raw/*.json")
    cnt = 0
    with open(OUT, "w", encoding="utf-8") as outf:
        for f in files:
            j = json.load(open(f, encoding="utf-8"))
            text = simple_clean(j.get("raw_text",""))
            rec = {
                "id": f"doc_{cnt}",
                "title": (text.splitlines()[0] if text else ""),
                "text": text,
                "source_url": j.get("url"),
                "fetched_at": j.get("fetched_at")
            }
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt += 1
    print("Wrote", cnt, "documents to", OUT)

if __name__ == "__main__":
    main()
