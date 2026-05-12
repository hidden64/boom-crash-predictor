"""Helpers métier autour de la sortie du modèle CRASH 500.

Aucun ordre n'est effectivement passé : ce module se contente de classifier
le signal IA pour que le frontend / les tests puissent l'exploiter.
Sur CRASH 500, le signal recommande de VENDRE/SHORTER (la chute attendue est
descendante).
"""
from dataclasses import dataclass


CRASH_ALERT_THRESHOLD = 0.80
CRASH_WARNING_THRESHOLD = 0.60


@dataclass
class TradingSignal:
    probability: float
    level: str  # "idle" | "warning" | "alert"
    should_sell: bool


def classify_signal(probability: float) -> TradingSignal:
    """Convertit une probabilité de crash [0,1] en signal métier prêt à servir."""
    prob = max(0.0, min(1.0, float(probability)))
    if prob >= CRASH_ALERT_THRESHOLD:
        return TradingSignal(probability=prob, level="alert", should_sell=True)
    if prob >= CRASH_WARNING_THRESHOLD:
        return TradingSignal(probability=prob, level="warning", should_sell=False)
    return TradingSignal(probability=prob, level="idle", should_sell=False)
