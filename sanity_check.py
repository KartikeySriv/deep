import torch
import torch.nn.functional as F

from model import GPTModel

from dataset import (
    data,
    get_batch,
    BATCH_SIZE,
    CONTEXT_LENGTH,
)

from tokenizer import VOCAB_SIZE


# ============================================================
# Configuration
# ============================================================

EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROP_RATE = 0.1

LEARNING_RATE = 3e-4

SANITY_STEPS = 150

PRINT_EVERY = 10

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
# Get ONE fixed batch
# ============================================================

x, y = get_batch(
    data=data,
    batch_size=BATCH_SIZE,
    context_length=CONTEXT_LENGTH,
)

x = x.to(device)
y = y.to(device)


print("\nFIXED BATCH")
print("-----------")

print("x shape:", x.shape)
print("y shape:", y.shape)


# ============================================================
# Create a FRESH model
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

model = model.to(device)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Loss function
# ============================================================

def calculate_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    batch_size, sequence_length, vocab_size = logits.shape

    # [B,T,V] → [B*T,V]
    logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size,
    )

    # [B,T] → [B*T]
    targets = targets.reshape(
        batch_size * sequence_length
    )

    loss = F.cross_entropy(
        logits,
        targets,
    )

    return loss


# ============================================================
# Initial loss
# ============================================================

model.train()

with torch.no_grad():

    initial_logits = model(x)

    initial_loss = calculate_loss(
        initial_logits,
        y,
    )

print("\nINITIAL LOSS")
print("------------")
print(f"{initial_loss.item():.4f}")


# ============================================================
# Train repeatedly on SAME batch
# ============================================================

print("\nSANITY CHECK")
print("------------")

for step in range(1, SANITY_STEPS + 1):

    # --------------------------------------------------------
    # 1. Clear gradients
    # --------------------------------------------------------

    optimizer.zero_grad(set_to_none=True)


    # --------------------------------------------------------
    # 2. Forward pass
    # --------------------------------------------------------

    logits = model(x)


    # --------------------------------------------------------
    # 3. Calculate loss
    # --------------------------------------------------------

    loss = calculate_loss(
        logits,
        y,
    )


    # --------------------------------------------------------
    # 4. Backpropagation
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # 5. Update parameters
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # 6. Print progress
    # --------------------------------------------------------

    if (
        step == 1
        or step % PRINT_EVERY == 0
        or step == SANITY_STEPS
    ):

        print(
            f"Step {step:3d}/{SANITY_STEPS} "
            f"| Loss: {loss.item():.6f}"
        )


# ============================================================
# Final loss
# ============================================================

print("\nSANITY CHECK COMPLETE")
print("---------------------")

print(
    f"Initial loss: {initial_loss.item():.6f}"
)

print(
    f"Final loss  : {loss.item():.6f}"
)

print(
    f"Loss reduction: "
    f"{initial_loss.item() - loss.item():.6f}"
)