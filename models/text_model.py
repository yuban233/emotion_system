import os
from typing import Dict

import torch


EMOTION_LABELS = ["angry", "happy", "sad", "neutral"]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "text_bert")

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
	global _tokenizer, _model

	if _tokenizer is not None and _model is not None:
		return _tokenizer, _model

	try:
		from transformers import AutoTokenizer, AutoModelForSequenceClassification
	except Exception as e:
		raise RuntimeError(f"transformers is not installed: {e}")

	if not os.path.isdir(MODEL_DIR):
		raise RuntimeError(
			f"Text BERT model not found at {MODEL_DIR}. "
			"Please train first: python train/train_text_bert.py"
		)

	_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
	_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
	_model.eval()
	return _tokenizer, _model


def predict_text_emotion(text: str) -> Dict:
	"""
	输入: 自然语言文本
	输出: {emotion, confidence, scores, error?}
	"""
	if not isinstance(text, str) or not text.strip():
		return {
			"emotion": "neutral",
			"confidence": 0.0,
			"scores": {label: 0.0 for label in EMOTION_LABELS},
			"error": "empty text",
		}

	try:
		tokenizer, model = _load_model()
		encoded = tokenizer(
			text,
			truncation=True,
			padding=True,
			max_length=128,
			return_tensors="pt",
		)
		encoded = {k: v.to(_device) for k, v in encoded.items()}

		with torch.no_grad():
			logits = model(**encoded).logits
			probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()

		best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
		scores = {label: round(float(probs[i]), 6) for i, label in enumerate(EMOTION_LABELS)}

		return {
			"emotion": EMOTION_LABELS[best_idx],
			"confidence": round(float(probs[best_idx]), 4),
			"scores": scores,
		}
	except Exception as e:
		return {
			"emotion": "neutral",
			"confidence": 0.0,
			"scores": {label: 0.0 for label in EMOTION_LABELS},
			"error": str(e),
		}
