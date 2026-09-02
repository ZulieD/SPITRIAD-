#!/usr/bin/env python3
"""
embed_ProtT5_from_csv.py
------------------------
Génère des embeddings ProtT5-XL-U50 per-residue (Lx1024) à partir d'un CSV
structuré avec au minimum les colonnes :
  - 'sequence'          : séquence acides aminés
  - 'esm_ProteinID'     : identifiant UniProt (utilisé comme clé dans le .h5)
    (alias acceptés : 'protein_id', 'biovec_UniProtID')

Usage:
    python embed_ProtT5_from_csv.py \\
        --csv  data/train.csv \\
        --out_dir results/prot_t5/ \\
        [--model_name Rostlab/prot_t5_xl_half_uniref50-enc] \\
        [--max_residues 4000] \\
        [--max_seq_len 1000] \\
        [--max_batch 100] \\
        [--skip_existing]

Sortie :
    <out_dir>/<csv_stem>_per_residue.h5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer


# ---------------------------------------------------------------------------
# Helpers CSV
# ---------------------------------------------------------------------------

ID_COLUMN_CANDIDATES = ["esm_ProteinID", "protein_id", "biovec_UniProtID"]


def load_sequences_from_csv(csv_path: str) -> dict:
    """
    Lit un CSV et retourne un dict {protein_id: sequence}.
    Cherche la colonne ID dans ID_COLUMN_CANDIDATES (premier trouvé).
    La colonne séquence est 'sequence'.
    Les acides aminés non-standard U/Z/O sont remplacés par X.
    Les gaps sont supprimés.
    """
    df = pd.read_csv(csv_path)

    # Trouver colonne ID
    id_col = None
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            f"Aucune colonne ID trouvée dans {csv_path}.\n"
            f"Colonnes disponibles : {list(df.columns)}\n"
            f"Colonnes acceptées   : {ID_COLUMN_CANDIDATES}"
        )

    if "sequence" not in df.columns:
        raise ValueError(f"Colonne 'sequence' introuvable dans {csv_path}.")

    # Dédoublonnage sur l'ID (garde la première occurrence)
    df = df.drop_duplicates(subset=[id_col])

    seqs = {}
    for _, row in df.iterrows():
        uid = str(row[id_col]).strip().replace("/", "_").replace(".", "_")
        seq = str(row["sequence"]).strip().upper()
        seq = seq.replace("-", "").replace("U", "X").replace("Z", "X").replace("O", "X")
        if uid and seq:
            seqs[uid] = seq

    print(f"  [{Path(csv_path).name}] {len(seqs)} séquences chargées (colonne ID : '{id_col}')")
    return seqs


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------


import os
import glob

def resolve_snapshot_path(model_path: str) -> str:
    """
    Prend le dossier de cache HF complet (ex: .../models--Rostlab--prot_t5_xl_half_uniref50-enc)
    et retourne le chemin réel du snapshot contenant config.json, sans accès réseau.
    """
    model_path = os.path.abspath(model_path)

    # Cas 1 : l'utilisateur a déjà donné le dossier snapshot directement
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path

    # Cas 2 : l'utilisateur a donné le dossier racine "models--namespace--name"
    snapshots_dir = os.path.join(model_path, "snapshots")
    if os.path.isdir(snapshots_dir):
        candidates = [
            d for d in glob.glob(os.path.join(snapshots_dir, "*"))
            if os.path.isfile(os.path.join(d, "config.json"))
        ]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            # Plusieurs snapshots : on prend le plus récemment modifié
            return max(candidates, key=os.path.getmtime)

    raise FileNotFoundError(
        f"Impossible de trouver config.json depuis le chemin fourni : {model_path}\n"
        f"Attendu soit un dossier snapshot direct, soit un dossier "
        f"'models--namespace--name' contenant un sous-dossier snapshots/<hash>/."
    )


def get_T5_model(model_path: str, device: torch.device):
    resolved_path = resolve_snapshot_path(model_path)
    print(f"Chargement du modèle depuis : {resolved_path} ...")

    model = T5EncoderModel.from_pretrained(
        resolved_path,
        local_files_only=True,
    ).eval().to(device)

    tokenizer = T5Tokenizer.from_pretrained(
        resolved_path,
        local_files_only=True,
    )

    print("  Modèle chargé avec succès.")
    return model, tokenizer

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def compute_per_residue_embeddings(
    model,
    tokenizer,
    seqs: dict,
    device: torch.device,
    max_residues: int = 4000,
    max_seq_len: int = 1000,
    max_batch: int = 100,
    already_done: set = None,
) -> dict:
    """
    Calcule les embeddings per-residue (Lx1024) pour toutes les séquences de `seqs`.
    Retourne un dict {protein_id: np.ndarray (L, 1024)}.
    `already_done` : ensemble d'IDs déjà présents dans le .h5 (pour --skip_existing).
    """
    if already_done is None:
        already_done = set()

    # Filtrer ceux déjà calculés
    seqs_to_run = {k: v for k, v in seqs.items() if k not in already_done}
    if len(seqs_to_run) < len(seqs):
        print(f"  {len(seqs) - len(seqs_to_run)} séquences déjà présentes → ignorées.")

    if not seqs_to_run:
        print("  Rien à calculer.")
        return {}

    # Tri par longueur décroissante (réduit le padding)
    seq_list = sorted(seqs_to_run.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_total  = len(seq_list)

    results = {}
    start   = time.time()
    batch   = []

    for seq_idx, (prot_id, seq) in enumerate(seq_list, 1):
        seq_len     = len(seq)
        seq_spaced  = " ".join(list(seq))
        batch.append((prot_id, seq_spaced, seq_len))

        n_res_batch = sum(s for _, _, s in batch) + seq_len
        flush = (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or seq_idx == n_total
            or seq_len > max_seq_len
        )
        if not flush:
            continue

        prot_ids, seqs_batch, seq_lens = zip(*batch)
        batch = []

        token_enc = tokenizer.batch_encode_plus(
            seqs_batch, add_special_tokens=True, padding="longest"
        )
        input_ids = torch.tensor(token_enc["input_ids"]).to(device)
        attn_mask = torch.tensor(token_enc["attention_mask"]).to(device)

        try:
            with torch.no_grad():
                out = model(input_ids, attention_mask=attn_mask)
        except RuntimeError as e:
            print(f"  RuntimeError pour batch autour de '{prot_ids[0]}' : {e}")
            continue

        for b_idx, pid in enumerate(prot_ids):
            s_len = seq_lens[b_idx]
            emb   = out.last_hidden_state[b_idx, :s_len]          # (L, 1024)
            results[pid] = emb.detach().cpu().float().numpy()      # float32

        if seq_idx % 100 == 0 or seq_idx == n_total:
            elapsed = time.time() - start
            rate    = seq_idx / max(elapsed, 1e-6)
            print(f"  {seq_idx}/{n_total} séquences traitées "
                  f"({elapsed/60:.1f} min, {rate:.1f} seq/s)")

    elapsed = time.time() - start
    avg     = elapsed / max(len(results), 1)
    print(f"\n  ✓ {len(results)} embeddings calculés en {elapsed/60:.1f} min "
          f"({avg:.3f} s/protéine)")
    return results


# ---------------------------------------------------------------------------
# Sauvegarde HDF5
# ---------------------------------------------------------------------------

def save_to_h5(emb_dict: dict, h5_path: str) -> None:
    """Ajoute (mode 'a') les embeddings dans un fichier HDF5."""
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    with h5py.File(h5_path, "a") as hf:
        for prot_id, emb in emb_dict.items():
            if prot_id in hf:
                del hf[prot_id]
            hf.create_dataset(prot_id, data=emb, compression="gzip", compression_opts=4)
    print(f"  Sauvegardé → {h5_path}  ({len(emb_dict)} nouvelles entrées)")


def get_existing_ids(h5_path: str) -> set:
    """Retourne l'ensemble des clés déjà présentes dans un .h5 (vide si absent)."""
    if not os.path.isfile(h5_path):
        return set()
    with h5py.File(h5_path, "r") as hf:
        return set(hf.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ProtT5 per-residue embeddings à partir d'un fichier CSV"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Fichier CSV d'entrée"
    )
    parser.add_argument(
        "--out_dir", default="./prot_t5_embeddings",
        help="Dossier de sortie pour le fichier .h5"
    )
    parser.add_argument(
        "--model_name", default="Rostlab/prot_t5_xl_half_uniref50-enc",
        help="Nom du modèle HuggingFace"
    )
    parser.add_argument("--max_residues", type=int, default=4000)
    parser.add_argument("--max_seq_len",  type=int, default=1000)
    parser.add_argument("--max_batch",    type=int, default=100)
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Ignorer les protéines déjà présentes dans le .h5 de sortie"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERREUR : fichier introuvable → {args.csv}")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU  : {props.name}")
        print(f"  VRAM : {props.total_memory / 1e9:.1f} GB")

    os.makedirs(args.out_dir, exist_ok=True)

    stem    = Path(args.csv).stem
    h5_path = os.path.join(args.out_dir, f"{stem}_per_residue.h5")

    print(f"\n{'='*60}")
    print(f"CSV      : {args.csv}")
    print(f"Sortie   : {h5_path}")
    print(f"{'='*60}")

    model, tokenizer = get_T5_model(args.model_name, device)

    seqs = load_sequences_from_csv(args.csv)
    if not seqs:
        print("Aucune séquence valide, abandon.")
        sys.exit(1)

    already_done = get_existing_ids(h5_path) if args.skip_existing else set()
    if already_done:
        print(f"  {len(already_done)} entrées déjà dans {h5_path}")

    emb_dict = compute_per_residue_embeddings(
        model, tokenizer, seqs, device,
        max_residues=args.max_residues,
        max_seq_len=args.max_seq_len,
        max_batch=args.max_batch,
        already_done=already_done,
    )

    if emb_dict:
        save_to_h5(emb_dict, h5_path)

    print(f"\n{'='*60}")
    print(f"Terminé. Fichier .h5 : {h5_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
