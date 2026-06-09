from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    direction: str
    final_score: float
    pa_score: float
    ml_prob: float
    sentiment_score: float
    confidence: float
    notes: str = ""
