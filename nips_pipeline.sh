#!/bin/bash --login

noise_level=100

# Training
python3 -u nips_main.py 30 $noise_level

# Inference
python3 -u nips_infer.py False $noise_level # The extract 45 is computed without multiprocess for memory usage reasons
python3 -u nips_infer.py True $noise_level

# Final score
python3 -u nips_final_score.py $noise_level

