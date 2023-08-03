# Google Research - Identify Contrails to Reduce Global Warming

## TODO:
1. Add file to download models - ✓
2. Add masks sizes column to the df
3. Stratify validation by masks sizes
4. Add an option to choose the scheduler
5. Write the inference part of the main.py

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