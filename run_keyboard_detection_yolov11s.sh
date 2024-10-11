#!/bin/bash

# Specify the environment name and staging path
ENVNAME=keyboard-detection
STAGING_PATH=/staging/groups/jacobucci_group

# Copy the packed Conda environment from staging
echo "Copying Conda environment from /staging..."
cp $STAGING_PATH/train_keyboard_detection.tar.gz ./

# Set the environment directory name (same as the ENVNAME unless specified otherwise)
export ENVDIR=$ENVNAME

# Handle setting up the environment path and activating it
echo "Setting up Conda environment..."
mkdir -p $ENVDIR
tar -xzf train_keyboard_detection.tar.gz -C $ENVDIR

# Activate the environment using an explicit path to Conda's activate script
echo "Activating Conda environment..."
source $ENVDIR/bin/activate  # Use 'source' instead of '.'

# Verify that the environment is activated and check the PATH variable
echo "Environment activated. Current PATH: $PATH"
which python  # This should show the path to the python executable inside the conda environment
python --version  # Verify the Python version to ensure it's from the Conda environment

# If 'which python' fails, attempt to manually add Python to PATH
if ! which python; then
    echo "Python not found after activation. Manually adding to PATH."
    export PATH=$ENVDIR/bin:$PATH
    echo "Updated PATH: $PATH"
    which python
fi

# Copy necessary dataset files from /staging
echo "Copying dataset files from /staging..."
cp $STAGING_PATH/train2017.zip ./
cp $STAGING_PATH/val2017.zip ./
cp $STAGING_PATH/annotations_trainval2017.zip ./

# Unzip the dataset files into the local working directory
echo "Unzipping dataset files..."
unzip -o train2017.zip -d train2017/
unzip -o val2017.zip -d val2017/
unzip -o annotations_trainval2017.zip -d annotations/

rm -rf train2017.zip val2017.zip annotations.zip

# Run the Python script for keyboard detection
echo "Running keyboard detection script..."
python3 keyboard_coco_chpc_yolov11s.py

# Clean up large files after job completes to avoid unnecessary file transfers
echo "Cleaning up extracted files..."

rm train_keyboard_detection.tar.gz

rm -rf train2017 val2017 annotations

echo "Deactivating environment and cleanup..."
conda deactivate