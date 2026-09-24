import torch
import torch.nn as nn

from tokenizer import encode, vocab_size


# ============================================================
# Configuration
# ============================================================

EMBEDDING_DIM = 32
MAX_CONTEXT_LENGTH = 8


# ============================================================
# Token + Position Embedding
# ============================================================

class InputEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_context_length: int
    ):
        super().__init__()

        # ----------------------------------------------------
        # Token embedding table 
        # Shape: [vocab_size, embedding_dim] [19*32]
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # ----------------------------------------------------
        # Position embedding table
        # Shape: [max_context_length, embedding_dim] [8*32]
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            max_context_length,
            embedding_dim
        )

        self.max_context_length = max_context_length


    # ========================================================
    # Forward pass
    # ========================================================

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        # input_ids shape:
        # [B, T]

        batch_size, sequence_length = input_ids.shape

        # ----------------------------------------------------
        # Make sure sequence isn't longer than our context
        # ----------------------------------------------------

        if sequence_length > self.max_context_length:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds maximum context length "
                f"{self.max_context_length}"
            )

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        token_embeddings = self.token_embedding(input_ids)

        # Shape:
        # [B, T, D]

        # ----------------------------------------------------
        # Position IDs
        # ----------------------------------------------------

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        # Shape:
        # [T]
        #
        # Example:
        # [0, 1, 2, 3, ...]

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        position_embeddings = self.position_embedding(positions)

        # Shape:
        # [T, D]

        # ----------------------------------------------------
        # Combine token + position information
        # ----------------------------------------------------

        x = token_embeddings + position_embeddings

        # Shape:
        # [B, T, D]

        return x


# ============================================================
# Test the complete Phase 2 pipeline
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Start with normal text
    # --------------------------------------------------------

    sample_text = "The dog"

    print("RAW TEXT")
    print("--------")
    print(sample_text)


    # --------------------------------------------------------
    # 2. Use Phase 1 tokenizer
    # --------------------------------------------------------

    token_ids = encode(sample_text)

    print("\nTOKEN IDs")
    print("---------")
    print(token_ids)


    # --------------------------------------------------------
    # 3. Convert list into PyTorch tensor
    # --------------------------------------------------------

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    print("\nINPUT IDS")
    print("---------")
    print(input_ids)

    print("Shape:", input_ids.shape)


    # --------------------------------------------------------
    # 4. Create embedding layer
    # --------------------------------------------------------

    embedding_layer = InputEmbedding(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        max_context_length=MAX_CONTEXT_LENGTH
    )


    # --------------------------------------------------------
    # 5. Forward pass
    # --------------------------------------------------------

    x = embedding_layer(input_ids)

    print("\nFINAL TRANSFORMER INPUT X")
    print("-------------------------")
    print(x)

    print("\nX shape:", x.shape)