# import torch
# import torch.nn.functional as F


def calculate_accuracy(logits, labels,ignore_index=-100):
    logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    labels = labels[:, 1:].contiguous().view(-1)
    values, indexes = logits.max(dim=-1)
    logits = indexes
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logits.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word