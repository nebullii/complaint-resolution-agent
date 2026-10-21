from datasets import load_dataset

DATASET = "liri-uzh/cfpb-complaints-mini"


def load_data(name=DATASET):
    """Return the frozen train and test splits as pandas DataFrames."""
    ds = load_dataset(name)
    train = ds["train"].to_pandas()
    test = ds["test"].to_pandas()
    return train, test
