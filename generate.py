import torch

from model import GPTModel

from tokenizer import (
    encode,
    decode,
    VOCAB_SIZE,
)

from embeddings import (
    EMBEDDING_DIM,
    CONTEXT_LENGTH,
)

from transformer_block import (
    NUM_HEADS,
    DROP_RATE,
)


# ============================================================
# Configuration
# ============================================================

NUM_LAYERS = 4

CHECKPOINT_PATH = "tiny_gpt_checkpoint.pth"

MAX_NEW_TOKENS = 50

SEED = 123


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DEVICE")
print("------")
print(device)


# ============================================================
# Create model
# ============================================================

model = GPTModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    context_length=CONTEXT_LENGTH,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    drop_rate=DROP_RATE,
    qkv_bias=False,
)


# ============================================================
# Load trained checkpoint
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)


# ============================================================
# Evaluation mode
# ============================================================

model.eval()


# ============================================================
# Generate text
# ============================================================

def generate_text(
    model: GPTModel,
    prompt: str,
    max_new_tokens: int,
) -> str:

    # --------------------------------------------------------
    # 1. Convert prompt → token IDs
    # --------------------------------------------------------

    token_ids = encode(prompt)

    # Convert to tensor with batch dimension:
    #
    # [T] → [1,T]

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )


    # --------------------------------------------------------
    # 2. No gradients during generation
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # ------------------------------------------------
            # Keep only latest context_length tokens
            # ------------------------------------------------

            input_context = input_ids[:, -CONTEXT_LENGTH:]


            # ------------------------------------------------
            # Run model
            # ------------------------------------------------

            logits = model(input_context)

            # Shape:
            #
            # [1,T,50257]


            # ------------------------------------------------
            # Take logits for final position
            # ------------------------------------------------

            last_logits = logits[:, -1, :]

            # Shape:
            #
            # [1,50257]


            # ------------------------------------------------
            # Greedy decoding
            # ------------------------------------------------

            next_token = torch.argmax(
                last_logits,
                dim=-1,
                keepdim=True,
            )

            # Shape:
            #
            # [1,1]


            # ------------------------------------------------
            # Append new token
            # ------------------------------------------------

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )


    # --------------------------------------------------------
    # 3. Convert token IDs back to text
    # --------------------------------------------------------

    generated_text = decode(
        input_ids[0].tolist()
    )

    return generated_text


# ============================================================
# Test generation
# ============================================================

if __name__ == "__main__":

    prompt = "The judge was"

    print("\nPROMPT")
    print("------")
    print(prompt)


    generated_text = generate_text(
        model=model,
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )


    print("\nGENERATED TEXT")
    print("--------------")
    print(generated_text)


    print("\nCHECKPOINT")
    print("----------")
    print(CHECKPOINT_PATH)

    print("\nMODEL CONFIGURATION")
    print("-------------------")
    print("Vocabulary size :", VOCAB_SIZE)
    print("Embedding dim   :", EMBEDDING_DIM)
    print("Context length  :", CONTEXT_LENGTH)
    print("Number of heads :", NUM_HEADS)
    print("Number of layers:", NUM_LAYERS)