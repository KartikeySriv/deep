import torch

from tokenizer import encode, decode


# ============================================================
# Configuration
# ============================================================

CONTEXT_LENGTH = 128
BATCH_SIZE = 8


# ============================================================
# Load and tokenize the complete dataset
# ============================================================

from pathlib import Path

DATA_PATH = Path("data/the-verdict.txt")

text = DATA_PATH.read_text(
    encoding="utf-8"
)

token_ids = encode(text)

# Convert the complete token sequence into a tensor
data = torch.tensor(
    token_ids,
    dtype=torch.long
)


# ============================================================
# Dataset information
# ============================================================

print("DATASET")
print("-------")
print("Characters:", len(text))
print("Tokens:", len(data))
print("Context length:", CONTEXT_LENGTH)
print("Batch size:", BATCH_SIZE)


# ============================================================
# Create one batch
# ============================================================

def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
):
    """
    Create a random batch of input-target pairs.

    X contains context_length tokens.
    Y is the same sequence shifted by one token.
    """

    # --------------------------------------------------------
    # Choose random starting positions
    # --------------------------------------------------------

    max_start = len(data) - context_length - 1

    start_positions = torch.randint(
        0,
        max_start + 1,
        (batch_size,)
    )

    # --------------------------------------------------------
    # Create input and target sequences
    # --------------------------------------------------------

    x = torch.stack(
        [
            data[start : start + context_length]
            for start in start_positions
        ]
    )

    y = torch.stack(
        [
            data[start + 1 : start + context_length + 1]
            for start in start_positions
        ]
    )

    return x, y


# ============================================================
# Test the batch creation
# ============================================================

if __name__ == "__main__":

    x, y = get_batch(
        data=data,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
    )

    print("\nBATCH SHAPES")
    print("------------")

    print("x shape:", x.shape)
    print("y shape:", y.shape)


    # --------------------------------------------------------
    # Display a few examples
    # --------------------------------------------------------

    print("\nBATCH EXAMPLES")
    print("--------------")

    for i in range(BATCH_SIZE):

        print(f"\nExample {i + 1}")

        print("Input IDs:")
        print(x[i])

        print("Target IDs:")
        print(y[i])

        print("\nInput text:")
        print(repr(
            decode(x[i].tolist())
        ))

        print("Target text:")
        print(repr(
            decode(y[i].tolist())
        ))