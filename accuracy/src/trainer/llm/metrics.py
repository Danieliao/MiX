"""
Metric for different llm tasks
"""

import torch

class Metric(object):
    def __init__(self):
        pass

class Perplexity(Metric):
    def __init__(self, seq_len:int=2048, stride:int=512):
        super().__init__()

        self.neg_log_likelihoods = []
        self.token_counts = []
        self.seq_len = seq_len
        self.stride = stride

    def func(self, pred:torch.Tensor, target:torch.Tensor):
        target = target.long()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss_val = loss_fn(pred.view(-1, pred.size(-1)), target.view(-1))
        return loss_val

    def update(self, pred:torch.Tensor, target:torch.Tensor):
        """
        pred: (batch, seq_len, vocab_size) - logits for tokens to score
        target: (batch, seq_len) - target token ids
        """
        n_tokens = target.numel()
        loss_val = self.func(pred, target)
        neg_log_likelihood = loss_val.float() * n_tokens
        self.neg_log_likelihoods.append(neg_log_likelihood)
        self.token_counts.append(n_tokens)

    def reduce(self):
        total_nll = torch.stack(self.neg_log_likelihoods).sum()
        total_tokens = sum(self.token_counts)
        ppl = torch.exp(total_nll / total_tokens)
        return ppl
