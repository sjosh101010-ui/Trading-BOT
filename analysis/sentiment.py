from transformers import pipeline
import numpy as np

_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        print("[sentiment] Loading FinBERT... (first run may take 30s)")
        _pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,
        )
    return _pipe


def score_headlines(headlines: list[str]) -> float:
    if not headlines:
        return 0.0

    pipe = get_pipeline()
    cleaned = [h[:512] for h in headlines[:20]]

    results = pipe(cleaned)
    scores = []
    for r in results:
        label = r["label"].lower()
        conf  = r["score"]
        if label == "positive":
            scores.append(conf)
        elif label == "negative":
            scores.append(-conf)
        else:
            scores.append(0.0)

    return round(float(np.mean(scores)), 4)


def sentiment_to_direction(score: float) -> str:
    if score > 0.3:
        return "POSITIVE"
    elif score < -0.3:
        return "NEGATIVE"
    else:
        return "NEUTRAL"
