import numpy as np

# EEG regions based on 10–20 system
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"]
}

def extract_eeg_features(df):
    features = {}

    # Region-wise statistics
    for region, channels in REGIONS.items():
        region_data = df[channels]

        features[f"{region}_mean"] = region_data.mean().mean()
        features[f"{region}_variance"] = region_data.var().mean()
        features[f"{region}_rms"] = np.sqrt((region_data ** 2).mean().mean())

    # Hemispheric asymmetry (left vs right)
    left_channels = ["Fp1", "F3", "F7", "T3", "C3", "T5", "P3", "O1"]
    right_channels = ["Fp2", "F4", "F8", "T4", "C4", "T6", "P4", "O2"]

    left_mean = df[left_channels].mean().mean()
    right_mean = df[right_channels].mean().mean()

    features["hemispheric_asymmetry"] = abs(left_mean - right_mean)

    return features
