# Google Research - Identify Contrails to Reduce Global Warming

## TODO:
1. Create a dataset - ✓
2. Create models - ✓
3. Create a training script - ✓
4. Add wandb to the train.py - ✓
4. Create configs and read them - ✓
5. Create validation in main.py - ✓
6. Do the train part of the main.py - ✓
7. Add train_fold to logging names - ✓
7. Add file to download models
7. Write the inference part of the main.py

## Project structure:

1. configs: config files with various params
2. dataset: dataset class for train/test
3. models: models code
4. notebooks: folder to store jupyter notebooks for tests
5. train: folder to store different training scripts
6. loss_functions: file for losses, but not needed anymore
7. main.py: run this file to train models or inference

## Run with:

```python main.py config.yaml --mode train```