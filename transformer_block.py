import torch
import torch.nn as nn

from attention import (
    MultiHeadCausalSelfAttention,
)

from feedforward import FeedForward

from embeddings import (
    TokenAndPositionEmbedding,
    EMBEDDING_DIM,
    CONTEXT_LENGTH,
)

from tokenizer import encode, VOCAB_SIZE


# ============================================================
# Configuration
# ============================================================

NUM_HEADS = 4
DROP_RATE = 0.1


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()

        # ----------------------------------------------------
        # LayerNorm before attention
        # ----------------------------------------------------

        self.ln1 = nn.LayerNorm(
            embedding_dim
        )

        # ----------------------------------------------------
        # Multi-Head Causal Self-Attention
        # ----------------------------------------------------

        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            qkv_bias=qkv_bias,
        )

        # ----------------------------------------------------
        # Dropout after attention
        # ----------------------------------------------------

        self.dropout1 = nn.Dropout(
            drop_rate
        )

        # ----------------------------------------------------
        # LayerNorm before feed-forward
        # ----------------------------------------------------

        self.ln2 = nn.LayerNorm(
            embedding_dim
        )

        # ----------------------------------------------------
        # Feed-Forward Network
        # ----------------------------------------------------

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim
        )

        # ----------------------------------------------------
        # Dropout after feed-forward
        # ----------------------------------------------------

        self.dropout2 = nn.Dropout(
            drop_rate
        )


    # ========================================================
    # Forward pass
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ====================================================
        # PART 1 — ATTENTION
        # ====================================================

        # Keep original input for residual connection
        shortcut = x

        # LayerNorm
        x = self.ln1(x)

        # Multi-head causal self-attention
        x = self.attention(x)

        # Dropout
        x = self.dropout1(x)

        # Residual connection
        x = x + shortcut


        # ====================================================
        # PART 2 — FEED FORWARD
        # ====================================================

        # Keep current representation for second residual
        shortcut = x

        # LayerNorm
        x = self.ln2(x)

        # Feed-forward network
        x = self.feed_forward(x)

        # Dropout
        x = self.dropout2(x)

        # Residual connection
        x = x + shortcut


        return x


# ============================================================
# Test Phase 1 → Phase 2 → Phase 3 → Phase 4
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
    # 2. Phase 1 — Tokenization
    # --------------------------------------------------------

    token_ids = encode(sample_text)

    print("\nTOKEN IDs")
    print("---------")
    print(token_ids)


    # --------------------------------------------------------
    # 3. Token IDs → tensor
    # --------------------------------------------------------

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
    )

    print("\nINPUT IDS")
    print("---------")
    print(input_ids)

    print("Shape:", input_ids.shape)


    # --------------------------------------------------------
    # 4. Phase 2 — Embeddings
    # --------------------------------------------------------

    embedding = TokenAndPositionEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_length=CONTEXT_LENGTH,
    )

    x = embedding(input_ids)

    print("\nEMBEDDING OUTPUT X")
    print("------------------")
    print("Shape:", x.shape)


    # --------------------------------------------------------
    # 5. Phase 4 — Transformer Block
    # --------------------------------------------------------

    block = TransformerBlock(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        context_length=CONTEXT_LENGTH,
        drop_rate=DROP_RATE,
        qkv_bias=False,
    )

    output = block(x)

    print("\nTRANSFORMER BLOCK OUTPUT")
    print("------------------------")
    print("Shape:", output.shape)


    # --------------------------------------------------------
    # 6. Configuration
    # --------------------------------------------------------

    print("\nCONFIGURATION")
    print("-------------")

    print(
        "Embedding dimension:",
        EMBEDDING_DIM
    )

    print(
        "Number of heads:",
        NUM_HEADS
    )

    print(
        "Head dimension:",
        EMBEDDING_DIM // NUM_HEADS
    )

    print(
        "Context length:",
        CONTEXT_LENGTH
    )

    print(
        "Dropout:",
        DROP_RATE
    )