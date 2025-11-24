#!/usr/bin/env python3

import os
import gzip
import pickle
import argparse
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from multiprocessing import Pool, cpu_count

from bin.eval.show_eval_metrics import show_metrics

from decifer.utility import extract_space_group_symbol, space_group_symbol_to_number

def rwp(sample, gen):
    """
    Calculates the residual (un)weighted profile between a sample and a generated XRD pattern
    """
    return np.sqrt(np.sum(np.square(sample - gen), axis=-1) / np.sum(np.square(sample), axis=-1))

def _extract_xrd_components(row: dict, label: str):
    """Extract continuous/discrete XRD arrays for ``label`` ("sample" or "gen")."""
    clean_entry = row.get(f'xrd_clean_{label}')
    q_continuous = iq_continuous = q_discrete = iq_discrete = None

    if isinstance(clean_entry, dict):
        q_continuous = clean_entry.get('q')
        iq_continuous = clean_entry.get('iq')
        q_discrete = clean_entry.get('q_disc')
        iq_discrete = clean_entry.get('iq_disc')

    # Fall back to flattened columns when the nested payload is missing.
    q_continuous = q_continuous if q_continuous is not None else row.get(f'xrd_q_continuous_{label}')
    iq_continuous = iq_continuous if iq_continuous is not None else row.get(f'xrd_iq_continuous_{label}')
    q_discrete = q_discrete if q_discrete is not None else row.get(f'xrd_q_discrete_{label}')
    iq_discrete = iq_discrete if iq_discrete is not None else row.get(f'xrd_iq_discrete_{label}')

    return q_continuous, iq_continuous, q_discrete, iq_discrete


def process_file(file_path):
    """Processes a single .pkl.gz file."""
    try:
        with gzip.open(file_path, 'rb') as f:
            row = pickle.load(f)

       # If successful generation, count 
        if 'success' not in row['status']:
            return None

        # Extract Validity
        formula_validity = row['validity']['formula']
        bond_length_validity = row['validity']['bond_length']
        spacegroup_validity = row['validity']['spacegroup']
        site_multiplicity_validity = row['validity']['site_multiplicity']
        valid = all([formula_validity, bond_length_validity, spacegroup_validity, site_multiplicity_validity])

        # Extract CIFs and XRD (Sample)
        cif_sample = row['cif_string_sample']
        (
            xrd_q_continuous_sample,
            xrd_iq_continuous_sample,
            xrd_q_discrete_sample,
            xrd_iq_discrete_sample,
        ) = _extract_xrd_components(row, 'sample')

        # Extract CIFs and XRD (Generated)
        cif_gen = row['cif_string_gen']
        (
            xrd_q_continuous_gen,
            xrd_iq_continuous_gen,
            xrd_q_discrete_gen,
            xrd_iq_discrete_gen,
        ) = _extract_xrd_components(row, 'gen')
        
        # Compute metrics only when XRD payloads exist.
        can_compute_wd = all(
            arr is not None
            for arr in (
                xrd_q_discrete_sample,
                xrd_q_discrete_gen,
                xrd_iq_discrete_sample,
                xrd_iq_discrete_gen,
            )
        )
        can_compute_rwp = (
            xrd_iq_continuous_sample is not None and xrd_iq_continuous_gen is not None
        )
        can_compute_l2 = can_compute_rwp

        if can_compute_wd:
            sample_weights = np.asarray(xrd_iq_discrete_sample, dtype=float)
            gen_weights = np.asarray(xrd_iq_discrete_gen, dtype=float)
            sample_weight_sum = np.sum(sample_weights)
            gen_weight_sum = np.sum(gen_weights)
            if sample_weight_sum > 0 and gen_weight_sum > 0:
                xrd_iq_discrete_sample_normed = sample_weights / sample_weight_sum
                xrd_iq_discrete_gen_normed = gen_weights / gen_weight_sum
                wd_value = wasserstein_distance(
                    np.asarray(xrd_q_discrete_sample, dtype=float),
                    np.asarray(xrd_q_discrete_gen, dtype=float),
                    u_weights=xrd_iq_discrete_sample_normed,
                    v_weights=xrd_iq_discrete_gen_normed,
                )
            else:
                wd_value = float('nan')
        else:
            wd_value = float('nan')

        # Rwp
        if can_compute_rwp:
            rwp_value = rwp(
                np.asarray(xrd_iq_continuous_sample, dtype=float),
                np.asarray(xrd_iq_continuous_gen, dtype=float),
            )
        else:
            rwp_value = float('nan')

        # L2 distance between continuous diffractograms
        if can_compute_l2:
            l2_distance = float(np.linalg.norm(
                np.asarray(xrd_iq_continuous_sample, dtype=float) - np.asarray(xrd_iq_continuous_gen, dtype=float)
            ))
        else:
            l2_distance = float('nan')

        # RMSD
        rmsd_value = row['rmsd']

        # Sequence lengths
        seq_len_sample = row['seq_len_sample']
        seq_len_gen = row['seq_len_gen']

        # Extract space group
        spacegroup_sym_sample = extract_space_group_symbol(cif_sample)
        spacegroup_num_sample = space_group_symbol_to_number(spacegroup_sym_sample)
        spacegroup_num_sample = int(spacegroup_num_sample) if spacegroup_num_sample is not None else np.nan

        spacegroup_sym_gen = extract_space_group_symbol(cif_gen)
        spacegroup_num_gen = space_group_symbol_to_number(spacegroup_sym_gen)
        spacegroup_num_gen = int(spacegroup_num_gen) if spacegroup_num_gen is not None else np.nan

        raw_match_mode = row.get('structure_match_mode')
        if isinstance(raw_match_mode, str):
            structure_match_mode = raw_match_mode.strip().lower() or "none"
        elif raw_match_mode is None:
            structure_match_mode = "none"
        elif isinstance(raw_match_mode, float) and np.isnan(raw_match_mode):
            structure_match_mode = "none"
        else:
            structure_match_mode = str(raw_match_mode).strip().lower() or "none"
        out_dict = {
            'index': row.get('index'),
            'cif_name': row.get('cif_name'),
            'rep': row.get('rep'),
            'dataset_name': row.get('dataset_name'),
            'model_name': row.get('model_name'),
            'rwp': rwp_value,
            'l2_distance': l2_distance,
            'wd': wd_value,
            'rmsd': rmsd_value,
            'rmsd_failure_cause': row.get('rmsd_failure_cause'),
            'cif_sample': cif_sample,
            'xrd_q_discrete_sample': xrd_q_discrete_sample,
            'xrd_iq_discrete_sample': xrd_iq_discrete_sample,
            'xrd_q_continuous_sample': xrd_q_continuous_sample,
            'xrd_iq_continuous_sample': xrd_iq_continuous_sample,
            'spacegroup_sym_sample': spacegroup_sym_sample,
            'spacegroup_num_sample': spacegroup_num_sample,
            'seq_len_sample': seq_len_sample,
            'cif_gen': cif_gen,
            'xrd_q_discrete_gen': xrd_q_discrete_gen,
            'xrd_iq_discrete_gen': xrd_iq_discrete_gen,
            'xrd_q_continuous_gen': xrd_q_continuous_gen,
            'xrd_iq_continuous_gen': xrd_iq_continuous_gen,
            'seq_len_gen': seq_len_gen,
            'spacegroup_sym_gen': spacegroup_sym_gen,
            'spacegroup_num_gen': spacegroup_num_gen,
            'formula_validity': formula_validity,
            'spacegroup_validity': spacegroup_validity,
            'bond_length_validity': bond_length_validity,
            'site_multiplicity_validity': site_multiplicity_validity,
            'validity': valid,
            'structure_match_mode': structure_match_mode,
        }
        return out_dict
    except Exception as e:
        raise e
        print(f"Error processing file {file_path}: {e}")
        return None

def process(folder, debug_max=None, top_k=None, top_k_metric: str = "rwp") -> pd.DataFrame:
    """Processes all files in the given folder using multiprocessing."""
    # Get list of files
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl.gz')]
    if debug_max is not None:
        files = files[:debug_max]

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing files..."))

    # Filter out None results and convert to DataFrame
    data_list = [res for res in results if res is not None]
    df = pd.DataFrame(data_list)

    if top_k is not None and top_k > 0 and not df.empty:
        metric_column = top_k_metric
        if metric_column == "l2":
            metric_column = "l2_distance"
        if metric_column not in df.columns:
            raise ValueError(f"Metric '{top_k_metric}' not available for sorting.")
        if 'index' in df.columns:
            group_column = 'index'
        elif 'cif_name' in df.columns:
            group_column = 'cif_name'
        else:
            raise ValueError(
                "Cannot apply per-sample top-k ranking because neither 'index' nor 'cif_name' is present in the data."
            )

        sort_columns = [group_column, metric_column]
        if 'rep' in df.columns:
            sort_columns.append('rep')
        df = (
            df.sort_values(by=sort_columns, ascending=True)
              .groupby(group_column, group_keys=False)
              .head(top_k)
        )

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-folder-paths", nargs='+', required=True, help="Provide a list of folder paths")
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help=(
            "Destination folder for collected pickles. "
            "Defaults to the parent directory of the first eval folder."
        ),
    )
    parser.add_argument("--debug_max", type=int, default=0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help=(
            "Rank rows per sample by the selected metric and keep the top-K entries "
            "per group (0 disables the ranking step)."
        ),
    )
    parser.add_argument(
        "--top-k-metric",
        choices=["rwp", "l2"],
        default="rwp",
        help="Metric used to rank rows when applying --top-k.",
    )
    args = parser.parse_args()
    if args.debug_max == 0:
        args.debug_max = None

    top_k = args.top_k if args.top_k > 0 else None

    # Create output folder; default to parent of first eval folder when not provided
    output_folder = args.output_folder
    if output_folder is None:
        # The eval folder is typically ".../eval_files/<dataset_name>"; we want to
        # drop the last component and write next to eval_files.
        output_folder = os.path.dirname(os.path.normpath(args.eval_folder_paths[0]))
        if output_folder == "":
            output_folder = "."
    os.makedirs(output_folder, exist_ok=True)

    # Loop over folders
    folder_names = [path.split("/")[-1] for path in args.eval_folder_paths]
    for label, path in zip(folder_names, args.eval_folder_paths):
        df = process(path, args.debug_max, top_k, args.top_k_metric)
        pickle_path = os.path.join(output_folder, label + '.pkl.gz')
        df.to_pickle(pickle_path)
        # RMSD values are exported as-is; any thresholding must be handled when computing metrics.
        show_metrics([pickle_path])
