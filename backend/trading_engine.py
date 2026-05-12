"""Helpers métier autour de la sortie du modèle.

Aucun ordre n'est effectivement passé : ce module se contente de classifier
le signal IA pour que le frontend / les tests puissent l'exploiter.
"""
from dataclasses import dataclass


SPIKE_ALERT_THRESHOLD = 0.80
SPIKE_WARNING_THRESHOLD = 0.60


@dataclass
class TradingSignal:
    probability: float
    level: str  # "idle" | "warning" | "alert"
    should_buy: bool


def classify_signal(probability: float) -> TradingSignal:
    """Convertit une probabilité [0,1] en signal métier prêt à servir."""
    prob = max(0.0, min(1.0, float(probability)))
    if prob >= SPIKE_ALERT_THRESHOLD:
        return TradingSignal(probability=prob, level="alert", should_buy=True)
    if prob >= SPIKE_WARNING_THRESHOLD:
        return TradingSignal(probability=prob, level="warning", should_buy=False)
    return TradingSignal(probability=prob, level="idle", should_buy=False)
