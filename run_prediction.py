#!/usr/bin/env python3
"""
predict_idr.py
==============
Inférence résidu-niveau IDR à partir de modèles pré-entraînés.
Reproduit EXACTEMENT la logique d'entraînement (hierarchical_idr_pipeline.py) :

  - MLP   : normalisation par source (sc["aa"], sc["t5"], sc["disto"]),
             pondération apprise par LearnedWeighter (gradient).
  - Sklearn: concat_weighted(X_aa, X_t5, X_disto, *grid_weights) issu du JSON,
             puis scaler global.

Task1 et Task2 sont indépendants et tournent sur la protéine ENTIÈRE (pas de
filtre par label).

Entrées requises :
  --meta        pipeline_meta.json   (produit lors de l'entraînement)
  --csv         CSV produit par fasta_to_input.py (colonnes : protein_id, sequence)
  --h5_aa       embeddings AAindex
  --h5_t5       embeddings ProtT5
  --h5_disto1   distogrammes pour task1
  --h5_disto2   distogrammes pour task2
  --out_dir     dossier de sortie

Sorties :
  <out_dir>/disorder/<pid>.caid   → p1  (struct vs disorder+binding)
  <out_dir>/binding/<pid>.caid    → p2  (disorder vs disorder-binding)
  <out_dir>/timings.csv

Format CAID :
  >P04637
  1\tM\t0.8923\t1
  2\tE\t0.3210\t0
  ...

timings.csv :
  # Running predict_idr.py, started Sun Feb  5 10:20:57 CET 2023
  sequence,milliseconds
  P04637,1827
"""

import argparse
import csv
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (identique à l'entraînement)
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_AAINDEX = 8
ENCODER_NAMES = [
    "alpha_structure", "beta_structure", "hydrophobicity", "volume",
    "energy", "hydropathy", "secondary_structure", "other_indexes",
]


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURES PYTORCH (identiques à l'entraînement)
# ─────────────────────────────────────────────────────────────────────────────

class LearnedWeighter(nn.Module):
    """Poids apprenables par gradient (softmax-normalisés)."""
    def __init__(self):
        super().__init__()
        self.log_w = nn.Parameter(torch.zeros(3))

    def forward(self, aa, t5, disto):
        w = torch.softmax(self.log_w, dim=0)
        return torch.cat([w[0] * aa, w[1] * t5, w[2] * disto], dim=-1)

    @property
    def weights(self):
        with torch.no_grad():
            return torch.softmax(self.log_w, dim=0).cpu().numpy()


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),                         nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 32),                          nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str) -> dict:
    """
    Charge un modèle MLP (.pt) ou sklearn (.pkl).
    La structure du dict retourné est identique à celle produite par
    pick_best_model() lors de l'entraînement.
    """
    path = str(path)
    if path.endswith(".pt"):
        payload  = torch.load(path, map_location=DEVICE, weights_only=False)
        dim_in   = payload["head_state"]["net.0.weight"].shape[1]
        weighter = LearnedWeighter().to(DEVICE)
        mlp      = FusionMLP(dim_in).to(DEVICE)
        weighter.load_state_dict(payload["weighter_state"])
        mlp.load_state_dict(payload["head_state"])
        weighter.eval()
        mlp.eval()
        # Reconstruction des StandardScaler depuis les arrays sérialisés
        sc = {}
        for k, d in payload["scalers"].items():
            s         = StandardScaler()
            s.mean_   = np.array(d["mean_"])
            s.scale_  = np.array(d["scale_"])
            sc[k]     = s
        return {
            "type":     "mlp",
            "weighter": weighter,
            "head":     mlp,
            "sc":       sc,          # {"aa": scaler, "t5": scaler, "disto": scaler}
        }
    else:
        p = joblib.load(path)
        return {
            "type":         "sklearn",
            "model":        p["model"],
            "scaler":       p["scaler"],      # scaler global (post-concat)
            "grid_weights": p["grid_weights"], # (w_aa, w_t5, w_disto) de la grille
        }


# ─────────────────────────────────────────────────────────────────────────────
# FUSION PONDÉRÉE (identique à l'entraînement)
# ─────────────────────────────────────────────────────────────────────────────

def concat_weighted(X_aa: np.ndarray, X_t5: np.ndarray, X_disto: np.ndarray,
                    w_aa: float, w_t5: float, w_disto: float) -> np.ndarray:
    """
    Concatène les 3 sources après normalisation par leurs poids respectifs.
    Même fonction qu'à l'entraînement.
    """
    s = w_aa + w_t5 + w_disto + 1e-9
    return np.concatenate(
        [(w_aa / s) * X_aa, (w_t5 / s) * X_t5, (w_disto / s) * X_disto],
        axis=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INFÉRENCE PAR TÂCHE
# ─────────────────────────────────────────────────────────────────────────────

def predict_task(model_info: dict,
                 X_aa: np.ndarray,
                 X_t5: np.ndarray,
                 X_disto: np.ndarray,
                 grid_weights: tuple | None = None) -> np.ndarray:
    """
    Retourne p ∈ [0, 1] pour chaque résidu, en reproduisant EXACTEMENT
    la logique utilisée lors de l'entraînement :

    MLP :
      1. Normalisation par source : sc["aa"], sc["t5"], sc["disto"]
      2. LearnedWeighter (poids appris) → concaténation pondérée
      3. FusionMLP → sigmoid

    Sklearn :
      1. concat_weighted(X_aa, X_t5, X_disto, *grid_weights)
         (grid_weights vient du JSON, identique à celui utilisé en train)
      2. scaler global → predict_proba[:, 1]

    Paramètre grid_weights : utilisé UNIQUEMENT pour sklearn ; pour MLP,
    les poids sont intégrés dans LearnedWeighter (état chargé depuis .pt).
    """
    if model_info["type"] == "mlp":
        sc       = model_info["sc"]
        weighter = model_info["weighter"]
        mlp      = model_info["head"]

        # Normalisation par source (identique à train_mlp_task)
        Xa = torch.FloatTensor(sc["aa"].transform(X_aa)).to(DEVICE)
        Xt = torch.FloatTensor(sc["t5"].transform(X_t5)).to(DEVICE)
        Xd = torch.FloatTensor(sc["disto"].transform(X_disto)).to(DEVICE)

        weighter.eval()
        mlp.eval()
        with torch.no_grad():
            return torch.sigmoid(mlp(weighter(Xa, Xt, Xd))).cpu().numpy()

    else:  # sklearn
        # grid_weights peut venir du model_info (pkl) ou du JSON (priorité JSON)
        gw = grid_weights if grid_weights is not None else model_info["grid_weights"]
        X  = model_info["scaler"].transform(
            concat_weighted(X_aa, X_t5, X_disto, *gw))
        return model_info["model"].predict_proba(X)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def _match_pid(pid: str, available_keys: set) -> str | None:
    """Même logique de normalisation que dans le pipeline d'entraînement."""
    for c in [
        pid,
        pid.replace("/", "_").replace(".", "_"),
        pid.replace("-", "_"),
        pid.strip(),
        pid.upper(),
        pid.lower(),
    ]:
        if c in available_keys:
            return c
    return None


def load_aaindex_h5(h5_path: str, protein_ids: list[str]) -> dict[str, np.ndarray]:
    """
    Structure : /proteins/<pid>/<encoder_name>  shape (L,)
    Retourne  : {pid: (L, 8) float32}
    """
    result = {}
    with h5py.File(h5_path, "r") as hf:
        root = hf.get("proteins", None)
        if root is None:
            print(f"  [WARN] AAindex : pas de groupe /proteins/ dans {h5_path}")
            return result
        avail = set(root.keys())
        for pid in protein_ids:
            key = _match_pid(pid, avail)
            if key is None:
                continue
            grp      = root[key]
            channels = [grp[n][:] for n in ENCODER_NAMES if n in grp]
            if len(channels) != N_AAINDEX:
                print(f"  [WARN] {pid} : {len(channels)}/{N_AAINDEX} encodeurs AAindex")
                continue
            result[pid] = np.stack(channels, axis=1).astype(np.float32)
    return result


def load_prott5_h5(h5_path: str, protein_ids: list[str]) -> dict[str, np.ndarray]:
    """
    Structure : /<pid>  shape (L, 1024)
    Retourne  : {pid: (L, 1024) float32}
    """
    result = {}
    with h5py.File(h5_path, "r") as hf:
        avail = set(hf.keys())
        for pid in protein_ids:
            key = _match_pid(pid, avail)
            if key is None:
                continue
            result[pid] = hf[key][:].astype(np.float32)
    return result


def load_distogram_h5(h5_path: str, protein_ids: list[str]) -> dict[str, np.ndarray]:
    """
    Structure : /proteins/<pid>/distogram_logits  shape (n_models, L, L, 64)
    Réduction : softmax → mean_j → mean_models  →  (L, 64)  [100% numpy/CPU]
    Retourne  : {pid: (L, 64) float32}
    """
    result = {}
    with h5py.File(h5_path, "r") as hf:
        root  = hf.get("proteins", hf)
        avail = set(root.keys())
        for pid in protein_ids:
            key = _match_pid(pid, avail)
            if key is None:
                continue
            grp = root[key]
            if "distogram_logits" not in grp:
                print(f"  [WARN] {pid} : clé 'distogram_logits' absente")
                continue
            logits = grp["distogram_logits"][()].astype(np.float32)
            # Même réduction que dans le pipeline d'entraînement
            logits -= logits.max(axis=-1, keepdims=True)   # stabilité numérique
            exp     = np.exp(logits)
            p       = exp / exp.sum(axis=-1, keepdims=True)  # softmax sur bins
            p       = p.mean(axis=2)                          # mean_j  → (n_models, L, 64)
            p       = p.mean(axis=0)                          # mean_models → (L, 64)
            result[pid] = p.astype(np.float32)
    return result


def load_csv_sequences(csv_path: str) -> dict[str, str]:
    """
    Charge le CSV produit par fasta_to_input.py.
    Colonnes attendues : protein_id, sequence
    Retourne : {protein_id: sequence}
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "protein_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError(
            f"Le CSV {csv_path} doit contenir les colonnes 'protein_id' et 'sequence'. "
            f"Colonnes trouvées : {list(df.columns)}"
        )
    return dict(zip(df["protein_id"].astype(str), df["sequence"].astype(str)))


# ─────────────────────────────────────────────────────────────────────────────
# ÉCRITURE DES FICHIERS CAID
# ─────────────────────────────────────────────────────────────────────────────

def write_caid(out_path: Path, pid: str, sequence: str,
               scores: np.ndarray, threshold: float) -> None:
    """
    Format :
      >P04637
      1\tM\t0.8923\t1
      2\tE\t0.3210\t0
      ...
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(f">{pid}\n")
        for i, (aa, score) in enumerate(zip(sequence, scores), start=1):
            label = 1 if score >= threshold else 0
            fh.write(f"{i}\t{aa}\t{score:.4f}\t{label}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prédiction IDR résidu-niveau — reproduit la logique d'entraînement")

    parser.add_argument("--meta",       required=True,
                        help="pipeline_meta.json produit lors de l'entraînement")
    parser.add_argument("--csv",        required=True,
                        help="CSV produit par fasta_to_input.py (colonnes : protein_id, sequence)")
    parser.add_argument("--h5_aa",      required=True,
                        help=".h5 embeddings AAindex")
    parser.add_argument("--h5_t5",      required=True,
                        help=".h5 embeddings ProtT5")
    parser.add_argument("--h5_disto1",  required=True,
                        help=".h5 distogrammes pour task1 (struct vs disorder+binding)")
    parser.add_argument("--h5_disto2",  required=True,
                        help=".h5 distogrammes pour task2 (disorder vs binding)")
    parser.add_argument("--out_dir",    default="results/",
                        help="Dossier de sortie (défaut: results/)")
    parser.add_argument("--threshold_disorder", type=float, default=0.5,
                        help="Seuil binarisation flavor 'disorder' (défaut: 0.5)")
    parser.add_argument("--threshold_binding",  type=float, default=0.5,
                        help="Seuil binarisation flavor 'binding'  (défaut: 0.5)")
    args = parser.parse_args()

    out_dir      = Path(args.out_dir)
    disorder_dir = out_dir / "disorder"
    binding_dir  = out_dir / "binding"
    disorder_dir.mkdir(parents=True, exist_ok=True)
    binding_dir.mkdir(parents=True, exist_ok=True)

    # ── Lecture du pipeline_meta.json ─────────────────────────────────────
    print(f"Lecture meta : {args.meta}")
    meta = json.loads(Path(args.meta).read_text())
    print(f"  task1 : type={meta['task1_type']}  path={meta['task1_path']}"
          f"  val_AUC={meta.get('task1_auc_val')}")
    print(f"  task2 : type={meta['task2_type']}  path={meta['task2_path']}"
          f"  val_AUC={meta.get('task2_auc_val')}")

    # Poids de grille depuis le JSON (source de vérité pour sklearn)
    # Pour MLP, ils ne sont pas utilisés (les poids sont dans le .pt)
    gw1 = tuple(meta.get("task1_grid_weights", [1/3, 1/3, 1/3]))
    gw2 = tuple(meta.get("task2_grid_weights", [1/3, 1/3, 1/3]))
    print(f"  grid_weights task1 : aa={gw1[0]:.3f}  t5={gw1[1]:.3f}  disto={gw1[2]:.3f}")
    print(f"  grid_weights task2 : aa={gw2[0]:.3f}  t5={gw2[1]:.3f}  disto={gw2[2]:.3f}")

    # ── Chargement des modèles ─────────────────────────────────────────────
    print(f"\nChargement modèle task1 : {meta['task1_path']}")
    model1 = load_model(meta["task1_path"])

    print(f"Chargement modèle task2 : {meta['task2_path']}")
    model2 = load_model(meta["task2_path"])

    # Affiche les poids appris si MLP
    for tid, m in [(1, model1), (2, model2)]:
        if m["type"] == "mlp":
            w = m["weighter"].weights
            print(f"  task{tid} MLP — poids LearnedWeighter : "
                  f"aa={w[0]:.3f}  t5={w[1]:.3f}  disto={w[2]:.3f}")

    # ── Chargement des séquences depuis le CSV ────────────────────────────
    sequences = load_csv_sequences(args.csv)
    pids      = list(sequences.keys())
    print(f"\n{len(pids)} protéine(s) dans le CSV")

    # ── Chargement des embeddings ──────────────────────────────────────────
    print("\nChargement des embeddings…")
    aa_dict = load_aaindex_h5(args.h5_aa,      pids)
    t5_dict = load_prott5_h5(args.h5_t5,       pids)
    d1_dict = load_distogram_h5(args.h5_disto1, pids)
    d2_dict = load_distogram_h5(args.h5_disto2, pids)
    print(f"  AAindex={len(aa_dict)}  ProtT5={len(t5_dict)}  "
          f"Disto1={len(d1_dict)}  Disto2={len(d2_dict)}")

    # ── Boucle de prédiction ───────────────────────────────────────────────
    run_start   = datetime.now()
    timing_rows: list[tuple[str, int]] = []

    print(f"\nPrédiction ({DEVICE}) …")
    n_ok = n_skip = 0

    for pid in pids:
        missing = []
        if pid not in aa_dict:  missing.append("AAindex")
        if pid not in t5_dict:  missing.append("ProtT5")
        if pid not in d1_dict:  missing.append("Disto_task1")
        if pid not in d2_dict:  missing.append("Disto_task2")
        if missing:
            print(f"  [SKIP] {pid} — embeddings manquants : {missing}")
            n_skip += 1
            continue

        seq = sequences[pid]
        t0  = time.time()

        # Longueur commune (sécurité)
        L = min(
            len(seq),
            aa_dict[pid].shape[0],
            t5_dict[pid].shape[0],
            d1_dict[pid].shape[0],
            d2_dict[pid].shape[0],
        )
        seq_trimmed = seq[:L]
        X_aa  = aa_dict[pid][:L]   # (L, 8)
        X_t5  = t5_dict[pid][:L]   # (L, 1024)
        X_d1  = d1_dict[pid][:L]   # (L, 64)  — distogramme dédié task1
        X_d2  = d2_dict[pid][:L]   # (L, 64)  — distogramme dédié task2

        # ── Task1 : struct vs disorder+binding  (protéine entière) ──────
        # MLP  → normalisation par source + LearnedWeighter (poids du .pt)
        # Sklearn → concat_weighted avec gw1 du JSON + scaler global
        p1 = predict_task(model1, X_aa, X_t5, X_d1, grid_weights=gw1)

        # ── Task2 : disorder vs disorder-binding  (protéine entière) ────
        p2 = predict_task(model2, X_aa, X_t5, X_d2, grid_weights=gw2)

        elapsed_ms = int((time.time() - t0) * 1000)
        timing_rows.append((pid, elapsed_ms))

        # ── Sorties CAID ─────────────────────────────────────────────────
        # disorder : p1  (probabilité d'être désordonné ou binding)
        write_caid(
            disorder_dir / f"{pid}.caid",
            pid, seq_trimmed, p1,
            threshold=args.threshold_disorder,
        )
        # binding : p2  (probabilité d'être disorder-binding)
        write_caid(
            binding_dir / f"{pid}.caid",
            pid, seq_trimmed, p2,
            threshold=args.threshold_binding,
        )

        print(f"  {pid}  L={L}  "
              f"disorder p1∈[{p1.min():.3f},{p1.max():.3f}]  "
              f"binding  p2∈[{p2.min():.3f},{p2.max():.3f}]  "
              f"{elapsed_ms} ms")
        n_ok += 1

    # ── timings.csv ────────────────────────────────────────────────────────
    timings_path = out_dir / "timings.csv"
    with open(timings_path, "w", newline="") as fh:
        fh.write(
            f"# Running predict_idr.py, started "
            f"{run_start.strftime('%a %b %d %H:%M:%S %Z %Y')}\n"
        )
        writer = csv.writer(fh)
        writer.writerow(["sequence", "milliseconds"])
        writer.writerows(timing_rows)

    print(f"\n{'='*60}")
    print(f"  Protéines prédites : {n_ok}   ignorées : {n_skip}")
    print(f"  disorder/   → {disorder_dir}")
    print(f"  binding/    → {binding_dir}")
    print(f"  timings.csv → {timings_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()