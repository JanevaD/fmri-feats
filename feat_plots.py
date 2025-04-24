import os
import numpy as np
import pandas as pd
import FeatureExtraction as fe
import plot as pl
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define paths
timeseries_root = '/home/danie/cnn/timeseries'
plot_output_root = '/home/danie/cnn/plots2'
results_csv = '/home/danie/cnn/Results k=2.csv'

os.makedirs(plot_output_root, exist_ok=True)

# Load group/cluster results
results = pd.read_csv(results_csv)
results.set_index("nident", inplace=True)

# Define regions to discard
exclude_regions = [
    "Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter", "Left-Cerebellum-White-Matter", "Right-Cerebellum-White-Matter", 
    "WM-hypointensities", "Optic-Chiasm", "CC_Posterior", "CC_Mid_Posterior", "CC_Central", "CC_Mid_Anterior", "CC_Anterior", 
    "LH_Background+FreeSurfer_Defined_Medial_Wall", "RH_Background+FreeSurfer_Defined_Medial_Wall",
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "Left-Inf-Lat-Vent", "Right-Inf-Lat-Vent", "3rd-Ventricle", "4th-Ventricle", 
    "Left-choroid-plexus", "Right-choroid-plexus", "CSF"
]

# Loop over each pipeline folder
for pipeline in os.listdir(timeseries_root):
    data_dir = os.path.join(timeseries_root, pipeline)
    if not os.path.isdir(data_dir):
        continue  # Skip if not a folder

    print(f"Processing pipeline: {pipeline}")

    plot_dir = os.path.join(plot_output_root, pipeline)
    os.makedirs(plot_dir, exist_ok=True)

    common_columns = None
    all_patients = []
    id_list = []

    # First pass to identify common columns across all valid files
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

        if len(df) < 150:
            print(f"Skipping {filename} (length: {len(df)})")
            continue

        if common_columns is None:
            common_columns = set(df.columns)
        else:
            common_columns.intersection_update(df.columns)

    if not common_columns:
        print(f"No valid time series found in {pipeline}. Skipping...")
        continue

    common_columns = list(common_columns - set(exclude_regions))

    # Second pass to process and extract features
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

        if len(df) < 150:
            continue

        try:
            id_value = int(filename.split("-")[1].split("_")[0])
        except Exception as e:
            print(f"Skipping file due to ID parse issue: {filename}")
            continue

        if id_value == 70:
            print(f"Skipping patient with ID 70: {filename}")
            continue

        id_list.append(id_value)
        time_series = df[common_columns]
        extractor = fe.FeatureExtraction(time_series)

        features = {
            "nident": id_value,
            "seg_fc": extractor.get_fc_seg_integ()[0],
            "integ_fc": extractor.get_fc_seg_integ()[1],
            "seg_mean_dfcs": extractor.get_mean_dfc()[0],
            "integ_mean_dfcs": extractor.get_mean_dfc()[1],
            "seg_var_dfcs": extractor.get_var_dfc()[0],
            "integ_var_dfcs": extractor.get_var_dfc()[1],
            "fluidity_var_intra": extractor.get_intra_fluidity_feats()[0],
            "fluidity_mean_intra": extractor.get_intra_fluidity_feats()[1],
            "fluidity_var_inter": extractor.get_inter_fluidity_feats()[0],
            "fluidity_mean_inter": extractor.get_inter_fluidity_feats()[1],
            "ms_plv": extractor.get_metastability_plv(),
            "ms_kop": extractor.get_metastability_kop(),
            "rsfa": extractor.get_rsfa(),
            "vars": extractor.get_vars(),
            "dvars": extractor.get_dvars(),
            "tvs": extractor.get_tv(),
            "ss": extractor.get_spectrogram_sum(),
            "entropies": extractor.get_entropies(),
            "alffs_mean": extractor.get_falff()[0],
            "alffs_var": extractor.get_falff()[1],
            "falffs_mean": extractor.get_falff()[2],
            "falffs_var": extractor.get_falff()[3],
        }

        all_patients.append(features)

    feature_names = list(all_patients[0].keys())
    feature_names.remove('nident')

# Define metastability feature names
    metastability_feats = ['ms_plv', 'ms_kop']

# Plot all features
    for f in feature_names:
        print(f"Plotting feature: {f}")
        if f in metastability_feats:
            # Plot metastability just once
            if f == 'ms_plv':  # Only trigger once to avoid duplicate plot
                pl.plot_metastability(all_patients, metastability_feats, results, pipeline, plot_dir)
        else:
            pl.plot_cluster_signif(all_patients, f, results, pipeline, plot_dir)