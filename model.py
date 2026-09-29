import torch
import torch.nn as nn

from tokenizer import VOCAB_SIZE, encode
from embeddings import (
    TokenAndPositionEmbedding,
    EMBEDDING_DIM,
    CONTEXT_LENGTH,
)
from transformer_block import (
    TransformerBlock,
    NUM_HEADS,
    DROP_RATE,
)


# ============================================================
# Model configuration
# ============================================================

NUM_LAYERS = 4


# ============================================================
# GPT Model
# ============================================================

class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        num_heads: int,
        num_layers: int,
        drop_rate: float,
        qkv_bias: bool = False,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 1. Token + Position Embeddings
        # ----------------------------------------------------

        self.embedding = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
        )

        # ----------------------------------------------------
        # 2. Stack Transformer Blocks
        # ----------------------------------------------------

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    drop_rate=drop_rate,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )

        # ----------------------------------------------------
        # 3. Final LayerNorm
        # ----------------------------------------------------

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        # ----------------------------------------------------
        # 4. Output Head
        #
        # [B,T,128]
        #     ↓
        # [B,T,50257]
        # ----------------------------------------------------

        self.out_head = nn.Linear(
            embedding_dim,
            vocab_size,
        )


    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        # input_ids:
        # [B,T]

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        x = self.embedding(input_ids)

        # [B,T,128]

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        for block in self.transformer_blocks:
            x = block(x)

        # [B,T,128]

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.final_norm(x)

        # [B,T,128]

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        logits = self.out_head(x)

        # [B,T,50257]

        return logits


# ============================================================
# Test the complete model
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
    # 2. Tokenization
    # --------------------------------------------------------

    token_ids = encode(sample_text)

    print("\nTOKEN IDs")
    print("---------")
    print(token_ids)


    # --------------------------------------------------------
    # 3. Convert token IDs to tensor
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
    # 4. Create GPT model
    # --------------------------------------------------------

    model = GPTModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_length=CONTEXT_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        drop_rate=DROP_RATE,
        qkv_bias=False,
    )


    # --------------------------------------------------------
    # 5. Forward pass
    # --------------------------------------------------------

    logits = model(input_ids)

    print("\nMODEL OUTPUT")
    print("------------")

    print("Logits shape:", logits.shape)


    # --------------------------------------------------------
    # 6. Inspect the last-token logits
    # --------------------------------------------------------

    last_logits = logits[:, -1, :]

    print("\nLAST TOKEN LOGITS")
    print("-----------------")

    print("Shape:", last_logits.shape)


    # --------------------------------------------------------
    # 7. Find the highest-scoring token
    # --------------------------------------------------------

    next_token_id = torch.argmax(
        last_logits,
        dim=-1,
    )

    print("\nPREDICTED NEXT TOKEN ID")
    print("-----------------------")
    print(next_token_id)


    # --------------------------------------------------------
    # 8. Decode prediction
    # --------------------------------------------------------

    from tokenizer import decode

    predicted_text = decode(
        next_token_id.tolist()
    )

    print("\nPREDICTED NEXT TOKEN")
    print("--------------------")
    print(repr(predicted_text))


    # --------------------------------------------------------
    # 9. Model configuration
    # --------------------------------------------------------

    print("\nMODEL CONFIGURATION")
    print("-------------------")

    print("Vocabulary size :", VOCAB_SIZE)
    print("Embedding dim   :", EMBEDDING_DIM)
    print("Context length  :", CONTEXT_LENGTH)
    print("Number of heads :", NUM_HEADS)
    print("Head dimension  :", EMBEDDING_DIM // NUM_HEADS)
    print("Number of layers:", NUM_LAYERS)
    print("Dropout         :", DROP_RATE)