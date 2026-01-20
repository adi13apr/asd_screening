import pandas as pd

CHANNELS = [
    "Fp1","Fp2","F3","F4","F7","F8",
    "T3","T4","C3","C4","T5","T6",
    "P3","P4","O1","O2",
    "Fz","Cz","Pz"
]

def load_eeg(csv_path):
    # Read without header
    df = pd.read_csv(csv_path, header=None)

    # Safety check
    if df.shape[1] != len(CHANNELS):
        raise ValueError(
            f"Expected {len(CHANNELS)} EEG channels, but found {df.shape[1]}"
        )

    # Assign correct channel names
    df.columns = CHANNELS

    return df
