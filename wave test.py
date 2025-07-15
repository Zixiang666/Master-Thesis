import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# !! IMPORTANT !!
# Point this path to the FOLDER containing the .hea and .dat files.
record_folder_path = '/Users/zixiang/ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1000/p10000032/s40689238'

# --- Logic to automatically discover the record file ---
path_for_wfdb = None
try:
    if not os.path.isdir(record_folder_path):
        raise FileNotFoundError(f"Error: Directory not found -> {record_folder_path}")

    for filename in os.listdir(record_folder_path):
        if filename.endswith('.hea'):
            base_name = filename.replace('.hea', '')
            path_for_wfdb = os.path.join(record_folder_path, base_name)
            break

    if path_for_wfdb is None:
        raise FileNotFoundError(f"Error: No .hea file found in the directory {record_folder_path}.")

    # --- Load the ECG Data ---
    print(f"Attempting to load record from path: {path_for_wfdb}")
    record = wfdb.rdrecord(path_for_wfdb)

    print(f"✅ Successfully loaded record: {record.record_name}")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Number of leads: {record.n_sig}")

    # --- Visualization logic for all leads ---
    signal_data = record.p_signal
    lead_names = record.sig_name
    units = record.units
    fs = record.fs
    num_leads = record.n_sig

    time_axis = np.arange(signal_data.shape[0]) / fs

    fig, axes = plt.subplots(num_leads, 1, figsize=(15, num_leads * 2), sharex=True)

    fig.suptitle(f'ECG Waveforms for Record: {os.path.basename(path_for_wfdb)}', fontsize=16)

    for i in range(num_leads):
        ax = axes[i]
        ax.plot(time_axis, signal_data[:, i])
        ax.set_ylabel(f'{lead_names[i]} ({units[i]})')
        ax.grid(True)

    axes[-1].set_xlabel('Time (seconds)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.show()

except Exception as e:
    print(f"❌ An error occurred: {e}")