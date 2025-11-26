#!/bin/bash

# Array of model epochs
epochs=(9 19 29 39)

# Boolean variables for dataset selection
asdiv=false
gsm8k=true
svamp=true

# Loop through each directory in cotqa
for dir in results/ballmer_20240726/cotqa/*/; do
    # Remove trailing slash from directory name
    dir=${dir%/}
    
    # Extract the directory name without the path
    dir_name=$(basename "$dir")
    
    # Check if the directory should be processed based on its name and boolean variables
    if ( $asdiv && [[ $dir_name == *"asdiv"* ]] ) || \
       ( $gsm8k && [[ $dir_name == *"gsm8k"* ]] ) || \
       ( $svamp && [[ $dir_name == *"svamp"* ]] ); then
        
        # Loop through each epoch
        for epoch in "${epochs[@]}"; do
            # Run the Python command
            python3 test_math_model.py \
                --results_dir "results/ballmer_20240726/cotqa/${dir_name}" \
                --batch_size 32 \
                --num_questions 400 \
                --model_epoch "$epoch"
            
            echo "Completed run for ${dir_name} with model_epoch ${epoch}"
        done
    else
        echo "Skipping ${dir_name} as it doesn't match the selected datasets"
    fi
done