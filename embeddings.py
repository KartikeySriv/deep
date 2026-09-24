import torch
import torch.nn as nn

from tokenizer import encode, decode, VOCAB_SIZE


# ============================================================
# Model configuration
# ============================================================

EMBEDDING_DIM = 128
CONTEXT_LENGTH = 128


# ============================================================
# Token + Position Embedding
# ============================================================

class TokenAndPositionEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int
    ):
        super().__init__()

        # ----------------------------------------------------
        # Token embedding table
        #
        # One row for every token in the vocabulary.
        #
        # Shape:
        # [vocab_size, embedding_dim]
        #
        # Here:
        # [50257, 128]
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # ----------------------------------------------------
        # Position embedding table
        #
        # One row for every possible position.
        #
        # Shape:
        # [context_length, embedding_dim]
        #
        # Here:
        # [128, 128]
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

        self.context_length = context_length


    # ========================================================
    # Forward pass
    # ========================================================

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        # input_ids:
        # [B, T]

        batch_size, sequence_length = input_ids.shape

        # ----------------------------------------------------
        # Make sure sequence fits inside context window
        # ----------------------------------------------------

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"context length ({self.context_length})"
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

        # Example:
        # [0, 1, 2, 3, ..., 127]

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        position_embeddings = self.position_embedding(
            positions
        )

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
# Test the Phase 1 → Phase 2 pipeline
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Raw text
    # --------------------------------------------------------

    sample_text = "The judge was"

    print("RAW TEXT")
    print("--------")
    print(sample_text)


    # --------------------------------------------------------
    # 2. Phase 1: BPE tokenization
    # --------------------------------------------------------

    token_ids = encode(sample_text)

    print("\nTOKEN IDs")
    print("---------")
    print(token_ids)


    # --------------------------------------------------------
    # 3. Convert token IDs to tensor
    # --------------------------------------------------------
    #
    # We add an outer list because the model expects:
    #
    # [B, T]
    #
    # Here B = 1
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
    # 4. Create embedding module
    # --------------------------------------------------------

    embedding = TokenAndPositionEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_length=CONTEXT_LENGTH
    )


    # --------------------------------------------------------
    # 5. Forward pass
    # --------------------------------------------------------

    x = embedding(input_ids)

    print("\nFINAL X")
    print("-------")
    print(x)

    print("\nX shape:", x.shape)


    # --------------------------------------------------------
    # 6. Print component shapes
    # --------------------------------------------------------

    print("\nEMBEDDING TABLE SHAPES")
    print("----------------------")

    print(
        "Token embedding table:",
        embedding.token_embedding.weight.shape
    )

    print(
        "Position embedding table:",
        embedding.position_embedding.weight.shape
    )