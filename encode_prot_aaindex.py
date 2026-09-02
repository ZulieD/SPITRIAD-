"""
encode_proteins_to_h5.py
========================
Lit un CSV contenant des séquences protéiques et les encode avec les
8 encodeurs de Medina-Ortiz et al. (2022), puis sauvegarde dans un .h5.

Usage:
    python encode_proteins_to_h5.py --input data.csv --output encodings.h5

Structure du HDF5 produit:
    /proteins/<protein_id>/
        sequence                   (str, attrs)
        alpha_structure            (float32, shape L)
        beta_structure             (float32, shape L)
        hydrophobicity             (float32, shape L)
        volume                     (float32, shape L)
        energy                     (float32, shape L)
        hydropathy                 (float32, shape L)
        secondary_structure        (float32, shape L)
        other_indexes              (float32, shape L)
        alpha_structure_fft        (float32, shape padded)
        beta_structure_fft         (float32, shape padded)
        ...etc pour chaque encodeur...
    /metadata/
        encoder_names              (liste des 8 noms)
        protein_ids                (liste des IDs)

Colonnes CSV attendues (au minimum):
    - sequence          : séquence en acides aminés
    - protein_disprot_id: identifiant unique de la protéine

Colonnes optionnelles conservées comme attributs:
    - residue_labels, cluster_fixed_dataset_label, protein_in_caid3
"""

import argparse
import ast
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 8 encodeurs (Table 1 — Medina-Ortiz et al. 2022)
# ---------------------------------------------------------------------------
ENCODERS = {
    "alpha_structure": {
        "A":  290.41, "R":  172.57, "N":  -38.37, "D":  159.43,
        "C":   -4.24, "Q": -268.55, "E":   -0.02, "G": -104.49,
        "H": -159.87, "I":  -34.08, "L":  -91.11, "K":  195.59,
        "M":   21.94, "F":   88.02, "P":  317.10, "S": -314.20,
        "T": -252.51, "W": -118.15, "Y":  -10.20, "V":  150.75,
    },
    "beta_structure": {
        "A":   71.85, "R":   -6.96, "N":  -90.14, "D":  -56.58,
        "C":   15.67, "Q":  -32.61, "E":   21.03, "G":  -62.33,
        "H":   31.27, "I":  164.64, "L":  -16.38, "K":   54.45,
        "M":  -18.77, "F":   21.61, "P":  115.37, "S": -106.56,
        "T":  -23.99, "W":  -76.02, "Y":  -15.49, "V":    9.929,
    },
    "hydrophobicity": {
        "A":    6.25, "R":   84.09, "N":  -21.73, "D":  -28.96,
        "C":  -34.88, "Q":   38.46, "E":  -21.48, "G":   53.16,
        "H":  -69.67, "I":  -54.85, "L":  -64.98, "K":  -52.92,
        "M":  -26.70, "F":  -21.46, "P":  -22.23, "S":   61.31,
        "T":   13.72, "W":   88.28, "Y":   40.85, "V":   33.77,
    },
    "volume": {
        "A":   44.65, "R":  200.15, "N": -191.18, "D": -232.26,
        "C": -156.21, "Q":  179.88, "E": -170.44, "G":  250.66,
        "H":  194.47, "I":  -88.56, "L": -201.08, "K": -118.84,
        "M": -227.61, "F":  -78.96, "P":  -44.80, "S":  221.12,
        "T":   -3.30, "W":   34.80, "Y":  203.07, "V":  184.45,
    },
    "energy": {
        "A": -107.79, "R":   51.15, "N":   73.94, "D":   55.36,
        "C":  -54.19, "Q":   31.44, "E":  -49.97, "G":   92.25,
        "H":  -39.54, "I":  -48.44, "L":    7.56, "K": -109.99,
        "M":   -7.39, "F":  -56.97, "P": -157.63, "S":  174.08,
        "T":   17.50, "W":  105.47, "Y":   36.61, "V":  -13.45,
    },
    "hydropathy": {
        "A":   15.33, "R":  172.36, "N": -259.13, "D": -216.01,
        "C": -242.01, "Q":  145.73, "E":    8.11, "G":  256.52,
        "H":  455.61, "I": -274.76, "L": -257.27, "K": -136.28,
        "M": -139.71, "F":   80.68, "P": -126.45, "S":  248.05,
        "T": -153.13, "W":   19.24, "Y":  171.61, "V":  231.50,
    },
    "secondary_structure": {
        "A":   56.16, "R":    1.44, "N":  -54.69, "D":  -29.38,
        "C":   10.07, "Q":  -15.43, "E":   20.20, "G":  -39.89,
        "H":   34.12, "I":   25.05, "L":  -10.20, "K":   55.31,
        "M":  -19.45, "F":   30.31, "P":   95.69, "S":  -85.57,
        "T":  -25.56, "W":  -59.91, "Y":   -4.25, "V":   15.99,
    },
    "other_indexes": {
        "A":   92.92, "R":  -37.39, "N":  -77.74, "D":   -7.42,
        "C":   40.04, "Q":  -45.52, "E":   50.74, "G":  -95.41,
        "H":   43.37, "I":   52.40, "L":    4.27, "K":   85.66,
        "M":   16.04, "F":   46.42, "P":  136.09, "S": -122.66,
        "T":  -31.46, "W": -124.49, "Y":  -33.07, "V":    7.21,
    },
}
ENCODER_NAMES = list(ENCODERS.keys())


# ---------------------------------------------------------------------------
# Fonctions d'encodage
# ---------------------------------------------------------------------------

def encode_sequence(sequence: str, table: dict, fill_value: float = 0.0) -> np.ndarray:
    """Encode une séquence AA avec un encodeur (dict AA -> float)."""
    return np.array(
        [table.get(aa.upper(), fill_value) for aa in sequence],
        dtype=np.float32,
    )


def apply_fft(vec: np.ndarray) -> np.ndarray:
    """
    Zero-pad à la prochaine puissance de 2 ≥ 2L-1, applique FFT,
    retourne le module complexe (float32).
    """
    L = len(vec)
    # prochaine puissance de 2 >= 2L - 1
    padded_len = 1
    while padded_len < (2 * L - 1):
        padded_len <<= 1
    padded = np.zeros(padded_len, dtype=np.float64)
    padded[:L] = vec
    spectrum = np.fft.fft(padded)
    return np.abs(spectrum).astype(np.float32)


def encode_protein(sequence: str) -> dict:
    """
    Retourne un dict avec les 8 encodages + les 8 spectres FFT.
    Clés: <encoder_name> et <encoder_name>_fft
    """
    result = {}
    for name, table in ENCODERS.items():
        vec = encode_sequence(sequence, table)
        result[name] = vec
        result[f"{name}_fft"] = apply_fft(vec)
    return result


# ---------------------------------------------------------------------------
# Parsing du CSV
# ---------------------------------------------------------------------------

def parse_residue_labels(raw) -> np.ndarray | None:
    """
    Convertit la colonne residue_labels (stockée comme string de liste)
    en tableau numpy int8. Retourne None si vide/invalide.
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    try:
        labels = ast.literal_eval(str(raw))
        return np.array(labels, dtype=np.int8)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Encode protein sequences from CSV into HDF5 using 8 AAIndex encoders."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Chemin vers le fichier CSV d'entrée",
    )
    parser.add_argument(
        "--output", "-o", default="protein_encodings.h5",
        help="Chemin vers le fichier HDF5 de sortie (défaut: protein_encodings.h5)",
    )
    parser.add_argument(
        "--id-col", default="protein_id",
        help="Colonne utilisée comme identifiant unique (défaut: protein_id)",
    )
    parser.add_argument(
        "--seq-col", default="sequence",
        help="Colonne contenant la séquence AA (défaut: sequence)",
    )
    parser.add_argument(
        "--no-fft", action="store_true",
        help="Ne pas calculer les spectres FFT (plus rapide, fichier plus léger)",
    )
    parser.add_argument(
        "--compression", default="gzip", choices=["gzip", "lzf", "none"],
        help="Compression HDF5 (défaut: gzip)",
    )
    args = parser.parse_args()

    # --- Lecture du CSV ---
    print(f"[1/4] Lecture du CSV : {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERREUR lors de la lecture du CSV : {e}", file=sys.stderr)
        sys.exit(1)

    # Vérification colonnes obligatoires
    missing = [c for c in [args.id_col, args.seq_col] if c not in df.columns]
    if missing:
        print(f"ERREUR : colonnes manquantes dans le CSV : {missing}", file=sys.stderr)
        print(f"Colonnes disponibles : {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Déduplique sur l'ID (garde la première occurrence)
    n_before = len(df)
    df = df.drop_duplicates(subset=[args.id_col])
    if len(df) < n_before:
        print(f"  ⚠  {n_before - len(df)} doublons supprimés (même {args.id_col})")

    # Supprime les lignes sans séquence
    df = df.dropna(subset=[args.seq_col])
    print(f"  → {len(df)} protéines à encoder")

    compression = None if args.compression == "none" else args.compression
    compression_opts = 4 if compression == "gzip" else None

    # --- Encodage + écriture HDF5 ---
    print(f"[2/4] Encodage et écriture dans : {args.output}")
    protein_ids = []

    with h5py.File(args.output, "w") as hf:
        grp_proteins = hf.create_group("proteins")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Encodage", unit="prot"):
            prot_id = str(row[args.id_col])
            sequence = str(row[args.seq_col]).strip().upper()

            if not sequence:
                print(f"  ⚠  {prot_id} : séquence vide, ignorée")
                continue

            protein_ids.append(prot_id)
            grp = grp_proteins.create_group(prot_id)

            # --- Attributs scalaires / métadonnées ---
            grp.attrs["sequence"] = sequence
            grp.attrs["length"] = len(sequence)

            # Colonnes optionnelles conservées comme attributs
            for col in ["cluster_fixed_dataset_label", "protein_in_caid3",
                        "FASTA", "protein_id", "biovec_UniProtID",
                        "alphafold_protein_id", "esm_ProteinID"]:
                if col in df.columns and not pd.isna(row.get(col)):
                    grp.attrs[col] = str(row[col])

            # --- Encodages des 8 représentations ---
            encodings = encode_protein(sequence)

            for name in ENCODER_NAMES:
                # Signal brut (L,)
                hf_kw = dict(compression=compression, compression_opts=compression_opts)
                grp.create_dataset(name, data=encodings[name], **hf_kw)

                # Spectre FFT (padded_length,) — optionnel
                if not args.no_fft:
                    grp.create_dataset(
                        f"{name}_fft",
                        data=encodings[f"{name}_fft"],
                        **hf_kw,
                    )

            # --- Labels résiduels si présents ---
            if "residue_labels" in df.columns:
                labels = parse_residue_labels(row.get("residue_labels"))
                if labels is not None:
                    grp.create_dataset(
                        "residue_labels",
                        data=labels,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                    grp.attrs["has_residue_labels"] = True

        # --- Métadonnées globales ---
        print("[3/4] Écriture des métadonnées globales")
        grp_meta = hf.create_group("metadata")

        # Liste des IDs en bytes pour HDF5
        dt_str = h5py.special_dtype(vlen=str)
        grp_meta.create_dataset(
            "protein_ids",
            data=np.array(protein_ids, dtype=object),
            dtype=dt_str,
        )
        grp_meta.create_dataset(
            "encoder_names",
            data=np.array(ENCODER_NAMES, dtype=object),
            dtype=dt_str,
        )
        grp_meta.attrs["n_proteins"] = len(protein_ids)
        grp_meta.attrs["n_encoders"] = len(ENCODER_NAMES)
        grp_meta.attrs["fft_included"] = not args.no_fft
        grp_meta.attrs["paper_doi"] = "10.3389/fmolb.2022.898627"
        grp_meta.attrs["source_csv"] = str(args.input)

    # --- Résumé ---
    print("[4/4] Résumé")
    output_path = Path(args.output)
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"  Protéines encodées : {len(protein_ids)}")
    print(f"  Encodeurs          : {len(ENCODER_NAMES)} (+ FFT = {not args.no_fft})")
    print(f"  Fichier HDF5       : {args.output}  ({size_mb:.2f} MB)")
    print("\nStructure HDF5 :")
    print("  /proteins/<id>/")
    for name in ENCODER_NAMES:
        print(f"      {name:<28} float32 (L,)")
        if not args.no_fft:
            print(f"      {name + '_fft':<28} float32 (padded,)")
    print("      residue_labels               int8    (L,)  [si présent dans CSV]")
    print("  /metadata/")
    print("      protein_ids, encoder_names, attrs")
    print("\nDone ✓")


# ---------------------------------------------------------------------------
# API Python (importable sans CLI)
# ---------------------------------------------------------------------------

def encode_csv_to_h5(
    csv_path: str,
    h5_path: str = "protein_encodings.h5",
    id_col: str = "protein_disprot_id",
    seq_col: str = "sequence",
    include_fft: bool = True,
    compression: str = "gzip",
) -> list[str]:
    """
    Version importable (sans argparse).

    Paramètres
    ----------
    csv_path    : chemin vers le CSV d'entrée
    h5_path     : chemin de sortie .h5
    id_col      : nom de la colonne identifiant
    seq_col     : nom de la colonne séquence
    include_fft : inclure les spectres FFT
    compression : 'gzip', 'lzf', ou 'none'

    Retourne
    --------
    Liste des protein_ids encodés
    """
    import sys as _sys
    # Patch argv pour éviter les conflits argparse si appelé depuis notebook
    _old_argv = _sys.argv
    _sys.argv = [
        "encode_proteins_to_h5.py",
        "--input", csv_path,
        "--output", h5_path,
        "--id-col", id_col,
        "--seq-col", seq_col,
        "--compression", compression,
    ]
    if not include_fft:
        _sys.argv.append("--no-fft")
    try:
        main()
    finally:
        _sys.argv = _old_argv


# ---------------------------------------------------------------------------
# Utilitaire de lecture rapide
# ---------------------------------------------------------------------------

def load_encoding(h5_path: str, protein_id: str, encoder: str = None,
                  fft: bool = False) -> dict | np.ndarray:
    """
    Charge l'encodage d'une protéine depuis le HDF5.

    Exemples
    --------
    >>> enc = load_encoding("encodings.h5", "DP04169")
    >>> enc["hydrophobicity"]          # array float32 shape (L,)
    >>> enc["hydrophobicity_fft"]      # spectre FFT

    >>> vec = load_encoding("encodings.h5", "DP04169", encoder="volume")
    >>> spec = load_encoding("encodings.h5", "DP04169", encoder="volume", fft=True)
    """
    with h5py.File(h5_path, "r") as hf:
        grp = hf[f"proteins/{protein_id}"]
        if encoder is not None:
            key = f"{encoder}_fft" if fft else encoder
            return grp[key][:]
        # Retourne tout
        result = {"sequence": grp.attrs["sequence"]}
        for key in grp.keys():
            result[key] = grp[key][:]
        return result


if __name__ == "__main__":
    main()