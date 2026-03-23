import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 4类标签顺序要和全系统保持一致
LABEL2ID = {"angry": 0, "happy": 1, "sad": 2, "neutral": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DATA_PATH = os.path.join("data", "text_emotion_4class.csv")
# 默认使用更小的中文BERT基座，先保证训练可跑通；可通过环境变量覆盖
BASE_MODEL = os.environ.get("TEXT_BASE_MODEL", "hfl/rbt3")
SAVE_DIR = os.path.join("models", "text_bert")
SEED = 42


@dataclass
class TextConfig:
    max_len: int = 128
    batch_size: int = 16
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01


CFG = TextConfig()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Please create CSV with columns: text,label where label in [angry,happy,sad,neutral]."
        )

    df = pd.read_csv(DATA_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: text,label")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(LABEL2ID.keys())]

    if len(df) < 200:
        raise ValueError("Text dataset too small. Please provide at least 200 labeled samples.")

    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def evaluate(model, loader, device):
    model.eval()
    preds, refs = [], []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            total_loss += float(loss.item()) * labels.size(0)
            n += labels.size(0)

            batch_preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            batch_refs = labels.detach().cpu().tolist()
            preds.extend(batch_preds)
            refs.extend(batch_refs)

    avg_loss = total_loss / max(1, n)
    acc = accuracy_score(refs, preds)
    f1 = f1_score(refs, preds, average="macro")
    return avg_loss, acc, f1, refs, preds


def save_eval_artifacts(y_true, y_pred):
    os.makedirs(SAVE_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(ID2LABEL.keys()))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[ID2LABEL[i] for i in range(4)],
        yticklabels=[ID2LABEL[i] for i in range(4)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Text Emotion Confusion Matrix")
    cm_path = os.path.join(SAVE_DIR, "text_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(4)],
        digits=4,
    )
    report_path = os.path.join(SAVE_DIR, "text_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved classification report: {report_path}")


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_data()
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label_id"],
    )

    print(f"Base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_ds = TextDataset(
        train_df["text"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        max_len=CFG.max_len,
    )
    val_ds = TextDataset(
        val_df["text"].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
        max_len=CFG.max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # 类别不平衡处理
    label_counts = np.bincount(train_df["label_id"].values, minlength=4).astype(np.float32)
    class_weights = (label_counts.sum() / (len(label_counts) * np.maximum(label_counts, 1.0))).astype(np.float32)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

    print("Train class distribution:", {ID2LABEL[i]: int(c) for i, c in enumerate(label_counts)})
    print("Class weights:", {ID2LABEL[i]: round(float(w), 4) for i, w in enumerate(class_weights.detach().cpu().numpy())})

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1.0
    best_pred = None

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for batch in train_loader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            n += labels.size(0)

        train_loss = total_loss / max(1, n)
        val_loss, val_acc, val_f1, y_true, y_pred = evaluate(model, val_loader, device)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_pred = (y_true, y_pred)
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"Epoch {epoch}: best model saved (f1_macro={val_f1:.4f})")

        print(
            f"Epoch {epoch}/{CFG.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
        )

    if best_pred is not None:
        save_eval_artifacts(best_pred[0], best_pred[1])

    print(f"Training finished. Best val_f1={best_f1:.4f}")
    print(f"Saved text BERT to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
