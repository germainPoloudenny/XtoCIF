#!/usr/bin/env python3
"""Liste les CIF qui échouent la vérification de bond-length validity."""

from __future__ import annotations

import argparse
import gzip
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

try:  # numpy est optionnel mais simplifie la normalisation des booléens.
    import numpy as np
except Exception:  # pragma: no cover - numpy est installé dans la plupart des configs.
    np = None  # type: ignore[assignment]


DEFAULT_EVAL_DIR = Path("runs/decifer/eval/noma/eval_files/default_dataset")


@dataclass
class Violation:
    file_path: Path
    cif_name: Optional[str]
    dataset_name: Optional[str]
    index: Optional[int]
    rep: Optional[int]
    reason: str
    cif_sample: Optional[str]
    cif_gen: Optional[str]


def _iter_eval_files(eval_dir: Path) -> Iterator[Path]:
    for path in sorted(eval_dir.glob("*.pkl.gz")):
        if path.is_file():
            yield path


def _load_row(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if np is not None:
        bool_types = (np.bool_,)  # type: ignore[attr-defined]
        int_types = (np.integer,)  # type: ignore[attr-defined]
        float_types = (np.floating,)  # type: ignore[attr-defined]
        if isinstance(value, bool_types):
            return bool(value)
        if isinstance(value, int_types):
            return bool(int(value))
        if isinstance(value, float_types):
            value = float(value)
    if isinstance(value, (int,)):
        return bool(value)
    if isinstance(value, float):
        if math.isnan(value):
            return False
        return bool(value)
    return bool(value)


def find_violations(files: Iterable[Path]) -> List[Violation]:
    violations: List[Violation] = []
    for file_path in files:
        try:
            row = _load_row(file_path)
        except Exception as exc:  # pragma: no cover - seulement journalisation.
            print(f"Impossible de lire {file_path}: {exc}", file=sys.stderr)
            continue

        validity = row.get("validity")
        if not isinstance(validity, dict):
            reason = "validity manquant"
            violations.append(
                Violation(
                    file_path=file_path,
                    cif_name=row.get("cif_name"),
                    dataset_name=row.get("dataset_name"),
                    index=row.get("index"),
                    rep=row.get("rep"),
                    reason=reason,
                    cif_sample=row.get("cif_string_sample"),
                    cif_gen=row.get("cif_string_gen"),
                )
            )
            continue

        bond_value = validity.get("bond_length")
        is_valid = _coerce_bool(bond_value) if bond_value is not None else False
        if is_valid:
            continue

        if bond_value is None:
            reason = "bond_length non renseigné"
        else:
            reason = f"bond_length={bond_value!r}"

        violations.append(
            Violation(
                file_path=file_path,
                cif_name=row.get("cif_name"),
                dataset_name=row.get("dataset_name"),
                index=row.get("index"),
                rep=row.get("rep"),
                reason=reason,
                cif_sample=row.get("cif_string_sample"),
                cif_gen=row.get("cif_string_gen"),
            )
        )

    return violations


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Affiche les CIFs dont la vérification « bond_length_validity » échoue "
            "dans les fichiers .pkl.gz produits par l'évaluation."
        )
    )
    parser.add_argument(
        "eval_dir",
        nargs="?",
        default=str(DEFAULT_EVAL_DIR),
        help="Dossier contenant les fichiers .pkl.gz (défaut: %(default)s).",
    )
    parser.add_argument(
        "--dump-cif",
        choices=["sample", "gen", "both"],
        help=(
            "Affiche les textes CIF associés aux violations (sample, gen ou les deux)."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limite le nombre de violations affichées/dumpées (0 = toutes).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    eval_dir = Path(args.eval_dir).expanduser()
    if not eval_dir.exists():
        print(f"Le dossier {eval_dir} est introuvable.", file=sys.stderr)
        return 2

    files = list(_iter_eval_files(eval_dir))
    total_files = len(files)
    if total_files == 0:
        print(f"Aucun fichier .pkl.gz trouvé dans {eval_dir}.")
        return 0

    violations = find_violations(files)
    if not violations:
        print(
            f"Aucune violation de bond-length détectée sur {total_files} fichier(s) dans {eval_dir}."
        )
        return 0

    ratio = len(violations) / total_files
    print(
        f"{len(violations)} fichier(s) sur {total_files} présentent une bond-length validity invalide "
        f"({ratio:.1%})."
    )
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else len(violations)
    for violation in violations[:max_rows]:
        cif_label = violation.cif_name or violation.file_path.stem
        dataset = violation.dataset_name or "inconnu"
        index = violation.index if violation.index is not None else "-"
        rep = violation.rep if violation.rep is not None else "-"
        print(
            f"- {cif_label} (dataset={dataset}, index={index}, rep={rep}) -> {violation.reason} "
            f"[{violation.file_path.name}]"
        )
        if args.dump_cif:
            if args.dump_cif in ("sample", "both"):
                print("  --- CIF SAMPLE ---")
                cif_text = violation.cif_sample.strip() if violation.cif_sample else "(indisponible)"
                print(cif_text)
            if args.dump_cif in ("gen", "both"):
                print("  --- CIF GEN ---")
                cif_text = violation.cif_gen.strip() if violation.cif_gen else "(indisponible)"
                print(cif_text)
            print()

    return 0


if __name__ == "__main__":  # pragma: no cover - point d'entrée script.
    raise SystemExit(main())
