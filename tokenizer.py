from pathlib import Path

import tiktoken


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/the-verdict.txt")

# GPT-2 compatible BPE tokenizer
ENCODER = tiktoken.get_encoding("gpt2")


# ============================================================
# Load dataset
# ============================================================

text = DATA_PATH.read_text(encoding="utf-8")


# ============================================================
# Tokenizer information
# ============================================================

VOCAB_SIZE = ENCODER.n_vocab


print("Tokenizer:", ENCODER.name)
print("Vocabulary size:", VOCAB_SIZE)
print("Text characters:", len(text))


# ============================================================
# Encode
# ============================================================

def encode(text: str) -> list[int]:
    """
    Convert text into GPT-2 BPE token IDs.
    """

    return ENCODER.encode(text)


# ============================================================
# Decode
# ============================================================

def decode(token_ids: list[int]) -> str:
    """
    Convert GPT-2 BPE token IDs back into text.
    """

    return ENCODER.decode(token_ids)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    sample_text = "The cat sat on the mat."

    token_ids = encode(sample_text)

    print("\nSample text:")
    print(sample_text)

    print("\nToken IDs:")
    print(token_ids)

    print("\nNumber of tokens:")
    print(len(token_ids))

    print("\nDecoded text:")
    print(decode(token_ids))

    full_token_ids = encode(text)

    print("\nFull dataset")
    print("------------")
    print("Characters:", len(text))
    print("Tokens:", len(full_token_ids))

    print("\nFirst 30 token IDs:")
    print(full_token_ids[:30])

    print("\nFirst 30 tokens decoded:")
    print(repr(decode(full_token_ids[:30])))