import argparse
import json
import os
from collections import Counter
import xml.etree.ElementTree as ET

import pandas as pd


TARGET_LABELS = ["angry", "happy", "sad", "neutral"]

# 尽量覆盖常见中英文/数字标签表达
LABEL_MAP = {
    "angry": "angry",
    "anger": "angry",
    "愤怒": "angry",
    "生气": "angry",
    "怒": "angry",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "like": "happy",
    "surprise": "happy",
    "喜悦": "happy",
    "开心": "happy",
    "高兴": "happy",
    "乐": "happy",

    "sad": "sad",
    "sadness": "sad",
    "fear": "sad",
    "悲伤": "sad",
    "难过": "sad",
    "沮丧": "sad",

    "neutral": "neutral",
    "none": "neutral",
    "中性": "neutral",
    "平静": "neutral",
    "一般": "neutral",
    "other": "neutral",
    "others": "neutral",
}

TEXT_COL_CANDIDATES = ["text", "content", "sentence", "utterance", "review", "评论"]
LABEL_COL_CANDIDATES = ["label", "emotion", "mood", "sentiment", "类别", "情绪"]


def normalize_label(raw):
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return LABEL_MAP.get(key)


def choose_col(columns, candidates):
    cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def parse_csv(path):
    df = None
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        return []

    text_col = choose_col(df.columns, TEXT_COL_CANDIDATES)
    label_col = choose_col(df.columns, LABEL_COL_CANDIDATES)
    if text_col is None or label_col is None:
        return []

    rows = []
    for _, row in df.iterrows():
        text = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
        label = normalize_label(row[label_col])
        if text and label in TARGET_LABELS:
            rows.append((text, label, os.path.basename(path)))
    return rows


def parse_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text_val = None
            label_val = None
            for k in TEXT_COL_CANDIDATES:
                if k in obj:
                    text_val = obj[k]
                    break
            for k in LABEL_COL_CANDIDATES:
                if k in obj:
                    label_val = obj[k]
                    break

            text = str(text_val).strip() if text_val is not None else ""
            label = normalize_label(label_val)
            if text and label in TARGET_LABELS:
                rows.append((text, label, os.path.basename(path)))
    return rows


def _safe_float(text):
    try:
        return float(text)
    except Exception:
        return 0.0


def map_cecps_sentence_to_label(sentence_elem):
    polarity_text = ""
    pol = sentence_elem.find("Polarity")
    if pol is not None and pol.text:
        polarity_text = pol.text.strip()

    joy = _safe_float((sentence_elem.findtext("Joy") or "0").strip())
    love = _safe_float((sentence_elem.findtext("Love") or "0").strip())
    expect = _safe_float((sentence_elem.findtext("Expect") or "0").strip())

    anger = _safe_float((sentence_elem.findtext("Anger") or "0").strip())
    hate = _safe_float((sentence_elem.findtext("Hate") or "0").strip())

    sorrow = _safe_float((sentence_elem.findtext("Sorrow") or "0").strip())
    anxiety = _safe_float((sentence_elem.findtext("Anxiety") or "0").strip())

    scores = {
        "angry": anger + hate,
        "happy": joy + love + expect,
        "sad": sorrow + anxiety,
    }

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if polarity_text == "中性":
        return "neutral"
    if best_score < 0.25:
        return "neutral"
    return best_label


def parse_cecps_xml(path):
    rows = []
    tree = ET.parse(path)
    root = tree.getroot()

    for sentence in root.iter("sentence"):
        text = (sentence.attrib.get("S") or "").strip()
        if not text:
            continue
        label = map_cecps_sentence_to_label(sentence)
        if label in TARGET_LABELS:
            rows.append((text, label, os.path.basename(path)))

    return rows


def load_records(input_dir):
    records = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            path = os.path.join(root, fname)
            lower = fname.lower()
            try:
                if lower.endswith(".csv"):
                    records.extend(parse_csv(path))
                elif lower.endswith(".jsonl"):
                    records.extend(parse_jsonl(path))
                elif lower.endswith(".xml"):
                    records.extend(parse_cecps_xml(path))
            except Exception as e:
                print(f"Skip {path}: {e}")
    return records


def balance_by_cap(df, cap_per_class):
    if cap_per_class <= 0:
        return df

    pieces = []
    for label in TARGET_LABELS:
        part = df[df["label"] == label]
        if len(part) > cap_per_class:
            part = part.sample(n=cap_per_class, random_state=42)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Build unified 4-class text emotion dataset.")
    parser.add_argument("--input-dir", default=os.path.join("data", "text_raw"), help="raw dataset directory")
    parser.add_argument("--output", default=os.path.join("data", "text_emotion_4class.csv"), help="output csv path")
    parser.add_argument("--cap-per-class", type=int, default=0, help="max samples per class, 0 means no cap")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(
            f"Input dir not found: {args.input_dir}. Put open-source text datasets (.csv/.jsonl) there first."
        )

    records = load_records(args.input_dir)
    if not records:
        raise RuntimeError("No valid samples found. Check file format and label mapping.")

    df = pd.DataFrame(records, columns=["text", "label", "source"])

    # 去重，优先保留首次出现
    df["text_norm"] = df["text"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["text_norm", "label"]).drop(columns=["text_norm"]).reset_index(drop=True)

    if args.cap_per_class > 0:
        df = balance_by_cap(df, args.cap_per_class)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved merged dataset to: {args.output}")
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    cnt = Counter(df["label"].tolist())
    for label in TARGET_LABELS:
        print(f"  {label}: {cnt.get(label, 0)}")


if __name__ == "__main__":
    main()
