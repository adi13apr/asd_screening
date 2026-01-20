from eeg_module import run_eeg_screening

if __name__ == "__main__":
    output = run_eeg_screening("data/sample_eeg.csv")
    print("\nEEG Screening Output:\n")
    print(output)
