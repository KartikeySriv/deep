import torch
import torch.nn as nn


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        expansion_factor: int = 4,
    ):
        super().__init__()

        hidden_dim = embedding_dim * expansion_factor

        # ----------------------------------------------------
        # Expand representation
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Non-linearity
        # ----------------------------------------------------

        self.gelu = nn.GELU()

        # ----------------------------------------------------
        # Project back to original dimension
        # ----------------------------------------------------

        self.fc2 = nn.Linear(
            hidden_dim,
            embedding_dim,
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x:
        # [B, T, D]

        x = self.fc1(x)

        # [B, T, 4D]

        x = self.gelu(x)

        x = self.fc2(x)

        # [B, T, D]

        return x


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    embedding_dim = 128

    x = torch.randn(
        1,
        3,
        embedding_dim,
    )

    feed_forward = FeedForward(
        embedding_dim=embedding_dim
    )

    output = feed_forward(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)