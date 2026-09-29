import torch
import torch.nn.functional as F

from model import GPTModel
from dataset import (
    data,
    get_batch,
    BATCH_SIZE,
    CONTEXT_LENGTH,
)

from tokenizer import (
    VOCAB_SIZE,
)


# ============================================================
# Configuration
# ============================================================

EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROP_RATE = 0.1

LEARNING_RATE = 3e-4

TRAIN_STEPS = 1000

PRINT_EVERY = 100

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

model = model.to(device)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# Parameter count
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nMODEL")
print("-----")

print("Total parameters:", total_parameters)
print("Total parameters (millions):",
      total_parameters / 1_000_000)


# ============================================================
# Loss calculation
# ============================================================

def calculate_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    # logits:
    # [B, T, V]

    # targets:
    # [B, T]

    batch_size, sequence_length, vocab_size = logits.shape

    # --------------------------------------------------------
    # Flatten logits
    # --------------------------------------------------------
    #
    # [B,T,V]
    #
    # becomes
    #
    # [B*T,V]
    # --------------------------------------------------------

    logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size,
    )

    # --------------------------------------------------------
    # Flatten targets
    # --------------------------------------------------------
    #
    # [B,T]
    #
    # becomes
    #
    # [B*T]
    # --------------------------------------------------------

    targets = targets.reshape(
        batch_size * sequence_length
    )

    # --------------------------------------------------------
    # Cross-entropy
    # --------------------------------------------------------

    loss = F.cross_entropy(
        logits,
        targets,
    )

    return loss


# ============================================================
# Training loop
# ============================================================

print("\nTRAINING")
print("--------")

model.train()


for step in range(1, TRAIN_STEPS + 1):

    # --------------------------------------------------------
    # 1. Get random training batch
    # --------------------------------------------------------

    x, y = get_batch(
        data=data,
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
    )

    x = x.to(device)
    y = y.to(device)


    # --------------------------------------------------------
    # 2. Clear previous gradients
    # --------------------------------------------------------

    optimizer.zero_grad(set_to_none=True)


    # --------------------------------------------------------
    # 3. Forward pass
    # --------------------------------------------------------

    logits = model(x)


    # --------------------------------------------------------
    # 4. Calculate loss
    # --------------------------------------------------------

    loss = calculate_loss(
        logits,
        y,
    )


    # --------------------------------------------------------
    # 5. Backpropagation
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # 6. Update model parameters
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # 7. Print progress
    # --------------------------------------------------------

    if step == 1 or step % PRINT_EVERY == 0:

        print(
            f"Step {step:4d}/{TRAIN_STEPS} "
            f"| Loss: {loss.item():.4f}"
        )


# ============================================================
# Save trained model
# ============================================================

checkpoint_path = "tiny_gpt_checkpoint.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": TRAIN_STEPS,
    },
    checkpoint_path,
)

print("\nTRAINING COMPLETE")
print("-----------------")
print("Checkpoint saved to:", checkpoint_path)