"""Utilitaires de nettoyage pour les CSV de ticks Deriv."""
import numpy as np
import pandas as pd


def clean_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """Élimine doublons, NaN et inf des ticks bruts. Retourne un DataFrame trié par timestamp."""
    if df is None or df.empty:
        return df

    df = df.drop_duplicates(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["timestamp", "price"]).reset_index(drop=True)
    return df
