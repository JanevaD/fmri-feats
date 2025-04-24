import os 
import numpy as np 
import pandas as pd 
import FeatureExtraction as fe
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_cluster_signif( all_patients, feat, results, pipeline, plot_dir):

    def remove_outliers_iqr(df):
        clean_df = pd.DataFrame()
        for feature in df['Feature'].unique():
            for cluster in df['Clusters'].unique():
                subset = df[(df['Feature'] == feature) & (df['Clusters'] == cluster)]
                q1 = subset['Value'].quantile(0.25)
                q3 = subset['Value'].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered = subset[(subset['Value'] >= lower_bound) & (subset['Value'] <= upper_bound)]
                clean_df = pd.concat([clean_df, filtered], axis=0)
        return clean_df

    nident = [patient["nident"] for patient in all_patients if "nident" in patient]
    all_features = pd.concat([patient[feat] for patient in all_patients if feat in patient], axis=1)
    all_features = all_features.transpose()
    all_features.index = nident
    all_features = all_features.merge(results, left_index=True, right_index=True)

    features_aware = all_features[all_features["Clusters"] == 0]   
    features_unaware = all_features[all_features["Clusters"] == 1]   

    # significance 
    signif= []
    p_values = []
    for feature in all_features.columns[:-1]:
        u_stat, p_value = mannwhitneyu(features_aware[feature], features_unaware[feature])
        signif.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})
        p_values.append(p_value)
    
    rejected, corrected_p_values, _, _ = multipletests(p_values, method = 'fdr_bh')
    for i, res in enumerate(signif):
        res["fdr-correctes-p"] = corrected_p_values[i]

    signif = pd.DataFrame(signif)

    features = pd.melt(all_features, id_vars='Clusters', var_name='Feature', value_name='Value')
    features = remove_outliers_iqr(features)

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ['#FF0101', '#0E0EFF']

    features_unique = features['Feature'].unique()
    clusters = features['Clusters'].unique()

    cluster_map = {cluster: idx for idx, cluster in enumerate(clusters)}
    features['Cluster_Num'] = features['Clusters'].map(cluster_map)

    for i, feature in enumerate(features_unique):
        for cluster, color in zip(clusters, colors):
            subset = features[(features['Feature'] == feature) & (features['Clusters'] == cluster)]
            x_positions = np.random.normal(i + cluster_map[cluster] * 0.3, 0.05, size=len(subset))  # Adjusted to give more space
            ax.scatter(x_positions, subset['Value'], color=color, alpha=0.6, s=40, label="Aware" if cluster == 'Aware' else "Unaware" if cluster == 'Unaware' else "")

    for i, feature in enumerate(features_unique):
        feature_data = [features[(features['Feature'] == feature) & (features['Clusters'] == cluster)]['Value'].values for cluster in clusters]
        parts = ax.boxplot(feature_data, positions=[i + cluster_map[cluster] * 0.3 for cluster in clusters],  # Adjusted boxplot positions
                            widths=0.3, patch_artist=True, showfliers=False)
        
        for patch, color in zip(parts['boxes'], colors):
            patch.set(facecolor='none', edgecolor=color, linewidth=1.5)

    ax.set_xticks(range(len(features_unique)))
    ax.set_xticklabels(features_unique, rotation=30, ha='right')
    ax.set_title('Feature Comparison by Clusters')
    ax.legend(title="Clusters")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    
    # Get the global max value for y-axis
    y_min, y_max = ax.get_ylim()
  
    
    # Define a fixed annotation line position based on the maximum y-value
    y_offset_text = (y_max - y_min) * 0.05  # 5% of the y-axis range
    y_annotation = y_max + y_offset_text  # Fixed line for annotations
    x_offset = 0.2  # Keeps the x-spacing fixed

    # Add significance annotations at the same line
    for i, feature in enumerate(features_unique):
        # Access p-value using .loc to match the feature name
        pval = signif.loc[signif['Feature'] == feature, 'p-value'].values[0]
        pval_cor = signif.loc[signif['Feature'] == feature, 'fdr-correctes-p'].values[0]

        pval_color = 'green' if pval < 0.05 else 'black'  # Highlight p-values less than 0.05 in green

        # Annotate p-value and corrected p-value on the fixed annotation line
        ax.annotate(f'p={pval:.3f}', xy=(i, y_annotation), ha='center', color=pval_color, fontsize=10)
        ax.annotate(f'pcor={pval_cor:.3f}', xy=(i, y_annotation - y_offset_text), ha='center', color='black', fontsize=10)

        # Draw centered horizontal line
        ax.hlines(y=y_annotation - y_offset_text / 2, xmin=i - x_offset, xmax=i + x_offset, color='black', linewidth=1)

    # Custom legend creation
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0101', markersize=10, alpha=0.4, label='Aware'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#0E0EFF', markersize=10, alpha=0.4, label='Unaware')]

    ax.legend(handles=legend_elements, title="Clusters")

    fig.suptitle(f"{pipeline}\n" f"{feat}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{pipeline}_{feat}.png"), dpi=300, bbox_inches='tight')
    plt.show()



def plot_metastability(all_patients, feat, results, pipeline, plot_dir):

    def remove_outliers_iqr(df):
        clean_df = pd.DataFrame()
        for feature in df['Feature'].unique():
            for cluster in df['Clusters'].unique():
                subset = df[(df['Feature'] == feature) & (df['Clusters'] == cluster)]
                q1 = subset['Value'].quantile(0.25)
                q3 = subset['Value'].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered = subset[(subset['Value'] >= lower_bound) & (subset['Value'] <= upper_bound)]
                clean_df = pd.concat([clean_df, filtered], axis=0)
        return clean_df

    nident = [patient["nident"] for patient in all_patients if "nident" in patient]

    feature_1_df = pd.concat([patient[feat[0]] for patient in all_patients if feat[0] in patient], axis=1).transpose()
    feature_1_df.index = nident

# Extract second feature (e.g., 'ms_kop') into another dataframe
    feature_2_df = pd.concat([patient[feat[1]] for patient in all_patients if feat[1] in patient], axis=1).transpose()
    feature_2_df.index = nident

# Combine both features side by side
    all_features = pd.concat([feature_1_df, feature_2_df], axis=1)
    all_features.columns = ['ms_plv', "ms_kop"]
    all_features = all_features.merge(results, left_index=True, right_index=True)
    
    features_aware = all_features[all_features["Clusters"] == 0]   
    features_unaware = all_features[all_features["Clusters"] == 1]   

    # significance 
    signif= []
    p_values = []
    for feature in all_features.columns[:-1]:
        u_stat, p_value = mannwhitneyu(features_aware[feature], features_unaware[feature])
        signif.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})
        p_values.append(p_value)
    
    p_values = np.ravel(p_values) 
    rejected, corrected_p_values, _, _ = multipletests(p_values, method = 'fdr_bh')
    for i, res in enumerate(signif):
        res["fdr-correctes-p"] = corrected_p_values[i]

    signif = pd.DataFrame(signif)

    features = pd.melt(all_features, id_vars='Clusters', var_name='Feature', value_name='Value')
    features = remove_outliers_iqr(features)

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ['#FF0101', '#0E0EFF']

    features_unique = features['Feature'].unique()
    clusters = features['Clusters'].unique()

    cluster_map = {cluster: idx for idx, cluster in enumerate(clusters)}
    features['Cluster_Num'] = features['Clusters'].map(cluster_map)

    for i, feature in enumerate(features_unique):
        for cluster, color in zip(clusters, colors):
            subset = features[(features['Feature'] == feature) & (features['Clusters'] == cluster)]
            x_positions = np.random.normal(i + cluster_map[cluster] * 0.3, 0.05, size=len(subset))  # Adjusted to give more space
            ax.scatter(x_positions, subset['Value'], color=color, alpha=0.6, s=40, label="Aware" if cluster == 'Aware' else "Unaware" if cluster == 'Unaware' else "")

    for i, feature in enumerate(features_unique):
        feature_data = [features[(features['Feature'] == feature) & (features['Clusters'] == cluster)]['Value'].values for cluster in clusters]
        parts = ax.boxplot(feature_data, positions=[i + cluster_map[cluster] * 0.3 for cluster in clusters],  # Adjusted boxplot positions
                            widths=0.3, patch_artist=True, showfliers=False)
        
        for patch, color in zip(parts['boxes'], colors):
            patch.set(facecolor='none', edgecolor=color, linewidth=1.5)

    ax.set_xticks(range(len(features_unique)))
    ax.set_xticklabels(features_unique, rotation=30, ha='right')
    ax.set_title('Feature Comparison by Clusters')
    ax.legend(title="Clusters")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    
    # Get the global max value for y-axis
    y_min, y_max = ax.get_ylim()
  
    
    # Define a fixed annotation line position based on the maximum y-value
    y_offset_text = (y_max - y_min) * 0.05  # 5% of the y-axis range
    y_annotation = y_max + y_offset_text  # Fixed line for annotations
    x_offset = 0.2  # Keeps the x-spacing fixed

    # Add significance annotations at the same line
    for i, feature in enumerate(features_unique):
        # Access p-value using .loc to match the feature name
        pval = signif.loc[signif['Feature'] == feature, 'p-value'].values[0]
        pval_cor = signif.loc[signif['Feature'] == feature, 'fdr-correctes-p'].values[0]

        pval_color = 'green' if pval < 0.05 else 'black'  # Highlight p-values less than 0.05 in green

        # Annotate p-value and corrected p-value on the fixed annotation line
        ax.annotate(f'p={pval:.3f}', xy=(i, y_annotation), ha='center', color=pval_color, fontsize=10)
        ax.annotate(f'pcor={pval_cor:.3f}', xy=(i, y_annotation - y_offset_text), ha='center', color='black', fontsize=10)

        # Draw centered horizontal line
        ax.hlines(y=y_annotation - y_offset_text / 2, xmin=i - x_offset, xmax=i + x_offset, color='black', linewidth=1)

    # Custom legend creation
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0101', markersize=10, alpha=0.4, label='Aware'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#0E0EFF', markersize=10, alpha=0.4, label='Unaware')]

    ax.legend(handles=legend_elements, title="Clusters")

    fig.suptitle(f"{pipeline}\n" f"{feat}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{pipeline}_{feat[0]}_{feat[1]}.png"), dpi=300, bbox_inches='tight')
    plt.show()
