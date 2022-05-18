import torch
from torch import nn, optim


def tune_temperature(logits, targs, *, verbose=True):
    """
    Tune the temperature of the model using the validation set.

    Thanks to:
    https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    criterion = nn.CrossEntropyLoss()

    # get labels and predictions
    logits = torch.Tensor(logits).to(torch.float32)
    targs = torch.Tensor(targs).to(torch.long)

    # calculate loss before temperature scaling
    loss_before = criterion(logits, targs).item()
    if verbose:
        print(f"Cross Entropy before temperature: {loss_before:.3f}")

    # optimize the temperature w.r.t. NLL
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def _eval():
        optimizer.zero_grad()
        # apply temperature scaling
        logits_ = logits / temperature.unsqueeze(1).expand(*logits.shape)
        loss = criterion(logits_, targs)
        loss.backward()
        return loss

    optimizer.step(_eval)

    # calculate loss after temperature scaling
    logits_ = logits / temperature.unsqueeze(1).expand(*logits.shape)
    loss_after = criterion(logits_, targs).item()
    temperature = temperature.item()
    if verbose:
        print(f"Optimal temperature: {temperature:.3f}")
        print(f"Cross Entropy after temperature: {loss_after:.3f}")

    return temperature
