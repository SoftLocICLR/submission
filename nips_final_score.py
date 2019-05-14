# Script performing the final score computation
# Mean of F1/pre/rec-scores across all test pieces

import numpy as np
import mir_eval
import pretty_midi
import sys

from nips_createDataset import *

# Display all individual statistics
bool_full_display = False

# Version
infer_run_name = "soft_model_" + str(sys.argv[1]) + "ms"
print("<<<<<", infer_run_name, ">>>>>")

# Gather all test samples
_, test_files = newSplitDataset()

# Inilialize placeholder for measures
scores = {'f1_mir': [], 'precision_mir': [], 'recall_mir': []}

for files in test_files:
    try:
        # Load prediction file
        prediction_list = np.load('infer/' + infer_run_name + "_" + files.split("/")[-1].replace('.wav', '.npy'))

        # Load ground-truth
        label_tmp = np.loadtxt(files.replace('.wav', '.txt'), skiprows=1)
        label_raw = label_tmp[:, [0, 2]]

        # Compute scores
        pre, rec, f1, _ = (
            mir_eval.transcription.precision_recall_f1_overlap(
                np.concatenate([label_raw[:, 0:1], label_raw[:, 0:1] + 1], axis=1),
                pretty_midi.note_number_to_hz(label_raw[:, 1]),
                np.concatenate([prediction_list[:, 0:1] - 0.005, prediction_list[:, 0:1] + 1], axis=1),
                pretty_midi.note_number_to_hz(prediction_list[:, 1]),
                offset_ratio=None))

        # Display individual scores
        if bool_full_display:
            print(f1, pre, rec)
            print(files.split("/")[-1])

        # Save measures
        scores['f1_mir'].append(f1)
        scores['precision_mir'].append(pre)
        scores['recall_mir'].append(rec)
    except:
        pass

# Make sure to have all predictions for definitive score
if len(scores['f1_mir']) != len(test_files):
    print('Caution only partial results! ', len(scores['f1_mir']), '/', len(test_files))

# Print out final score
print(np.mean(scores['f1_mir']), np.mean(scores['precision_mir']), np.mean(scores['recall_mir']), " (F1/prec/rec)")
