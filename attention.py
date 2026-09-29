import math

import torch
import torch.nn as nn

from tokenizer import encode, VOCAB_SIZE
from embeddings import (
    TokenAndPositionEmbedding,
    EMBEDDING_DIM,
    CONTEXT_LENGTH,
)


# ============================================================
# Model configuration
# ============================================================

NUM_HEADS = 4


# ============================================================
# Multi-Head Causal Self-Attention
# ============================================================

class MultiHeadCausalSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        qkv_bias: bool = False,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Check that embedding dimension can be divided
        # equally among all attention heads.
        # ----------------------------------------------------

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be "
                f"divisible by num_heads ({num_heads})"
            )

        # ----------------------------------------------------
        # Store dimensions
        # ----------------------------------------------------

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length

        # ----------------------------------------------------
        # Q / K / V projections
        #
        # Input:
        # [B, T, D]
        #
        # Output:
        # [B, T, D]
        #
        # qkv_bias=False matches the project configuration.
        # ----------------------------------------------------

        self.Wq = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=qkv_bias,
        )

        self.Wk = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=qkv_bias,
        )

        self.Wv = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=qkv_bias,
        )

        # ----------------------------------------------------
        # Output projection
        #
        # After combining all heads:
        #
        # [B, T, D]
        #
        # stays:
        #
        # [B, T, D]
        # ----------------------------------------------------

        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # ----------------------------------------------------
        # Causal mask
        #
        # Lower triangular matrix:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        #
        # True  = allowed
        # False = blocked
        # ----------------------------------------------------

        causal_mask = torch.tril(
            torch.ones(
                context_length,
                context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
        )


    # ========================================================
    # Forward pass
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):

        # x:
        # [B, T, D]

        batch_size, sequence_length, embedding_dim = x.shape

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) "
                f"exceeds context length "
                f"({self.context_length})"
            )

        # ----------------------------------------------------
        # 1. Create Q / K / V
        # ----------------------------------------------------

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # All:
        # [B, T, D]

        # ----------------------------------------------------
        # 2. Split into multiple heads
        # ----------------------------------------------------
        #
        # Before:
        #
        # [B, T, D]
        #
        # We reshape:
        #
        # [B, T, H, head_dim]
        #
        # Then transpose:
        #
        # [B, H, T, head_dim]
        # ----------------------------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Q, K, V:
        # [B, H, T, head_dim]

        # ----------------------------------------------------
        # 3. Calculate QK^T
        # ----------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # Shape:
        # [B, H, T, T]

        # ----------------------------------------------------
        # 4. Scale the attention scores
        # ----------------------------------------------------

        scores = scores / math.sqrt(self.head_dim)

        # ----------------------------------------------------
        # 5. Apply causal mask
        # ----------------------------------------------------

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        # ----------------------------------------------------
        # 6. Softmax
        # ----------------------------------------------------

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        # Shape:
        # [B, H, T, T]

        # ----------------------------------------------------
        # 7. Weighted sum of Values
        # ----------------------------------------------------

        context = weights @ V

        # Shape:
        # [B, H, T, head_dim]

        # ----------------------------------------------------
        # 8. Combine all heads
        # ----------------------------------------------------
        #
        # Current:
        #
        # [B, H, T, head_dim]
        #
        # Transpose:
        #
        # [B, T, H, head_dim]
        #
        # Merge H and head_dim:
        #
        # [B, T, D]
        # ----------------------------------------------------

        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        # ----------------------------------------------------
        # 9. Output projection
        # ----------------------------------------------------

        output = self.out_proj(context)

        # Shape:
        # [B, T, D]

        # ----------------------------------------------------
        # Optional debugging information
        # ----------------------------------------------------

        if return_attention:

            debug = {
                "Q": Q,
                "K": K,
                "V": V,
                "scores": scores,
                "weights": weights,
                "context": context,
            }

            return output, debug

        return output


# ============================================================
# Test Phase 1 → Phase 2 → Phase 3
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
    # 3. Convert IDs to tensor
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
    # 4. Phase 2: Token + Position Embeddings
    # --------------------------------------------------------

    embedding = TokenAndPositionEmbedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_length=CONTEXT_LENGTH,
    )

    x = embedding(input_ids)

    print("\nX")
    print("-")
    print("Shape:", x.shape)


    # --------------------------------------------------------
    # 5. Phase 3: Multi-Head Causal Self-Attention
    # --------------------------------------------------------

    attention = MultiHeadCausalSelfAttention(
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        context_length=CONTEXT_LENGTH,
        qkv_bias=False,
    )

    output, debug = attention(
        x,
        return_attention=True,
    )


    # --------------------------------------------------------
    # 6. Print shapes
    # --------------------------------------------------------

    print("\nATTENTION SHAPES")
    print("----------------")

    print("X:", x.shape)

    print("Q:", debug["Q"].shape)
    print("K:", debug["K"].shape)
    print("V:", debug["V"].shape)

    print("Scores:", debug["scores"].shape)
    print("Weights:", debug["weights"].shape)

    print("Context:", debug["context"].shape)

    print("Output:", output.shape)


    # --------------------------------------------------------
    # 7. Show causal attention weights
    # --------------------------------------------------------

    print("\nHEAD 0 ATTENTION WEIGHTS")
    print("------------------------")

    print(debug["weights"][0, 0])


    # --------------------------------------------------------
    # 8. Configuration
    # --------------------------------------------------------

    print("\nCONFIGURATION")
    print("-------------")

    print("Embedding dimension:", EMBEDDING_DIM)
    print("Number of heads:", NUM_HEADS)
    print("Head dimension:", EMBEDDING_DIM // NUM_HEADS)
    print("Context length:", CONTEXT_LENGTH)