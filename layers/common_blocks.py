import torch.nn as nn


class FlattenHead(nn.Module):
    def __init__(
        self,
        n_vars,
        nf,
        target_window,
        head_dropout=0,
        mode="linear",
        head_dropout_position="post",
    ):
        super().__init__()
        if head_dropout_position not in {"none", "pre", "post"}:
            raise ValueError(f"Unsupported head_dropout_position: {head_dropout_position}")
        self.n_vars = n_vars
        self.head_dropout_position = head_dropout_position
        self.flatten = nn.Flatten(start_dim=-2)
        if mode == "linear":
            self.head = nn.Linear(nf, target_window)
        else:
            self.head = nn.Sequential(
                nn.Linear(nf, nf // 2),
                nn.SiLU(),
                nn.Linear(nf // 2, target_window),
            )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        if self.head_dropout_position == "pre":
            x = self.dropout(x)
            return self.head(x)

        x = self.head(x)
        if self.head_dropout_position == "post":
            return self.dropout(x)
        return x
