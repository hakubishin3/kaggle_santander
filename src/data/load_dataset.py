import pandas as pd
from pathlib import Path


def get_dataset_filename(config, dataset_type: str):
    path = config['dataset']['input_directory']
    path += config['dataset']['files'][dataset_type]

    return Path(path)


def load_dataset(train_path, test_path, debug_mode: bool):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if debug_mode is True:
        train = train.iloc[:1000]
        test = test.iloc[:1000]

    return train, test
