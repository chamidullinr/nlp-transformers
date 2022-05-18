import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.preprocessing import LabelEncoder

from . import translation_utils
from .metrics import jaccard_index, levenshtein_score, text_accuracy


def concat_datasets(datasets1, datasets2):
    out = DatasetDict()
    for (split1, x), (split2, y) in zip(datasets1.items(), datasets2.items()):
        out[split1] = concatenate_datasets([x, y])
    return out


def _add_column_to_dataset(dataset: Dataset, name, value):
    if isinstance(value, (list, np.ndarray)):
        out = dataset.add_column(name, value)
    else:
        n = dataset.num_rows
        out = dataset.add_column(name, [value] * n)
    return out


def add_column(datasets: [Dataset, DatasetDict], name, value):
    if isinstance(datasets, Dataset):
        out = _add_column_to_dataset(datasets, name, value)
    elif isinstance(datasets, DatasetDict):
        out = DatasetDict()
        for ds_name, dataset in datasets.items():
            out[ds_name] = _add_column_to_dataset(dataset, name, value)
    return out


def encode_column(datasets: [Dataset, DatasetDict], name):
    def encode_records(records, encoder):
        records[name] = encoder.transform(records[name])
        return records

    # fit encoder
    encoder = LabelEncoder()
    if isinstance(datasets, Dataset):
        encoder.fit(datasets[name])
    elif isinstance(datasets, DatasetDict):
        encoder.fit(list(datasets.values())[0][name])

    encoded_datasets = datasets.map(
        encode_records, batched=True, fn_kwargs=dict(encoder=encoder)
    )

    return encoded_datasets, encoder


def create_predictions_df(
    tokenizer,
    dataset: Dataset,
    prediction_out,
    *,
    trg_feature="trg",
    metrics=[text_accuracy, levenshtein_score, jaccard_index]
):
    """Create DataFrame with predictions and targets. And evaluate metrics."""
    assert hasattr(prediction_out, "predictions")

    # create test dataframe with preds
    df = dataset.to_pandas()
    df["pred"] = tokenizer.batch_decode(
        prediction_out.predictions, skip_special_tokens=True
    )
    if hasattr(prediction_out, "probs"):
        df["prob"] = translation_utils.get_combined_probs(
            prediction_out, skip_prob_tok=-1
        )
        df["prob_ext"] = translation_utils.get_combined_probs_between_separators(
            prediction_out, tokenizer
        )

    # compute metrics
    for met in metrics:
        df[met.__name__] = df.apply(lambda r: met(r["pred"], r[trg_feature]), axis=1)

    return df
