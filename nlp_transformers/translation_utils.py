import numpy as np

from .models.model_outputs import TranslationOutput


def get_combined_probs(prediction_out: TranslationOutput, skip_prob_tok=-1):
    """Combine probabilities from prediction output ignoring skip prob token."""
    assert prediction_out.probs is not None
    probs = prediction_out.probs.copy()
    probs[probs == skip_prob_tok] = np.nan
    return np.nanmean(probs, 1)  # nanprod


def get_combined_probs_between_separators(prediction_out: TranslationOutput, tokenizer):
    """
    Method splits predicted sequences by separator (pipe character - "|")
    and returns combined probability of substrings between pipes.
    """
    assert prediction_out.probs is not None

    # get pipe ("|") token ids (pipe character can be in multiple tokens)
    pipe_token_ids = [v for k, v in tokenizer.vocab.items() if '|' in k]

    # apply loop for each record
    combined_probs_all = []
    for i in range(len(prediction_out.predictions)):
        pred = prediction_out.predictions[i][1:]
        prob = prediction_out.probs[i]

        # remove special tokens
        cond = ~np.isin(pred, tokenizer.all_special_ids)
        pred, prob = pred[cond], prob[cond]

        # get pipe token indices
        pipe_idxs = np.where(np.isin(pred, pipe_token_ids))[0].tolist()
        pipe_idxs = [0] + pipe_idxs + [len(pred)]

        # get combined probs separated by pipe ("|")
        combined_probs = [np.nanmean(prob[(prev_idx+1 if prev_idx != 0 else prev_idx):idx])  # nanprod
                          for prev_idx, idx in zip(pipe_idxs[:-1], pipe_idxs[1:])]
        combined_probs_all.append(combined_probs)

    return combined_probs_all
