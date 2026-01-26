# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path

# Durée maximale d'un trou que l'on accepte de combler (minutes)
MAX_GAP_MIN = 60.0


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "02_Data"

    path_in = data_dir / "PQ_BV_Cloutasse.csv"
    path_out = data_dir / "PQ_BV_Cloutasse_interp.csv"

    print(f"[INFO] Lecture : {path_in}")
    if not path_in.exists():
        raise FileNotFoundError(path_in)

    # Lecture fichier original
    df = pd.read_csv(
        path_in,
        sep=";",
        na_values=["NA", "NaN", "", "-9999"]
    )

    # Conversion des dates + tri temporel
    if "dateP" in df.columns:
        df["dateP"] = pd.to_datetime(df["dateP"])
    if "dateQ" not in df.columns:
        raise KeyError("Colonne 'dateQ' absente du fichier.")
    df["dateQ"] = pd.to_datetime(df["dateQ"])

    df = df.sort_values("dateQ").reset_index(drop=True)

    # Conversion Q_ls en numérique
    if "Q_ls" not in df.columns:
        raise KeyError("Colonne 'Q_ls' absente du fichier.")

    df["Q_ls"] = pd.to_numeric(df["Q_ls"], errors="coerce")

    # Pas de temps moyen (en minutes)
    dt_sec = (
        df["dateQ"].sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
        .median()
    )
    dt_min = float(dt_sec / 60.0)
    print(f"[INFO] Pas de temps moyen dt ≈ {dt_min:.2f} min")

    # Nombre maximum de pas consécutifs que l'on accepte de combler
    max_gap_steps = int(MAX_GAP_MIN / dt_min)
    max_gap_steps = max(max_gap_steps, 1)
    print(f"[INFO] Trous comblés jusqu'à {MAX_GAP_MIN:.1f} min "
          f"(≈ {max_gap_steps} pas consécutifs)")

    # On met l'index temporel pour une interpolation "time" propre
    df = df.set_index("dateQ")

    # Copie de la série brute pour stats
    q_raw = df["Q_ls"].copy()

    # Interpolation linéaire dans le temps, sans extrapolation
    # 1) interpolation vers l'avant (ne remplit pas le début si NaN)
    q_fwd = q_raw.interpolate(
        method="time",
        limit=max_gap_steps,
        limit_direction="forward"
    )
    # 2) interpolation vers l'arrière (ne remplit pas la fin si NaN)
    q_filled = q_fwd.interpolate(
        method="time",
        limit=max_gap_steps,
        limit_direction="backward"
    )

    # On garde les trous trop longs ou les bords comme NaN
    df["Q_ls_filled"] = q_filled

    # Quelques stats
    n_tot = len(q_raw)
    n_nan_before = q_raw.isna().sum()
    n_nan_after = df["Q_ls_filled"].isna().sum()
    n_filled = n_nan_before - n_nan_after

    print(f"[INFO] Points totaux           : {n_tot}")
    print(f"[INFO] NaN avant interpolation : {n_nan_before}")
    print(f"[INFO] NaN après interpolation : {n_nan_after}")
    print(f"[INFO] Points effectivement comblés : {n_filled}")

    # On remet dateQ comme colonne normale
    df = df.reset_index()

    # Réorganisation des colonnes :
    # on insère Q_ls_filled juste AVANT Q_ls, en gardant tout le reste
    cols = list(df.columns)
    new_cols = []
    for c in cols:
        if c == "Q_ls":
            # d'abord la colonne interpolée, puis l'originale
            if "Q_ls_filled" in df.columns and "Q_ls_filled" not in new_cols:
                new_cols.append("Q_ls_filled")
            new_cols.append("Q_ls")
        elif c != "Q_ls_filled":
            # on évite de rajouter Q_ls_filled deux fois
            new_cols.append(c)

    df = df[new_cols]

    # On écrit un fichier de sortie : mêmes colonnes, même ordre,
    # sauf que Q_ls_filled se glisse juste avant Q_ls
    df.to_csv(path_out, sep=";", index=False)
    print(f"[OK] Fichier interpolé écrit : {path_out}")


if __name__ == "__main__":
    main()
