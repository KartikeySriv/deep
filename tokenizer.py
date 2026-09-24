from pathlib import Path


# ============================================================
# 1. Load raw text
# ============================================================

text_path = Path("data/input.txt")

text = text_path.read_text(encoding="utf-8")

print("RAW TEXT")
print("--------")
print(text)


# ============================================================
# 2. Build vocabulary
# ============================================================

# Every unique character becomes a token.
# sorted() gives us a deterministic ordering.
vocab = sorted(set(text))

vocab_size = len(vocab)

print("\nVOCABULARY")
print("----------")
print(vocab)

print("\nVocabulary size:", vocab_size)


# ============================================================
# 3. Character -> Token ID
# ============================================================

stoi = {
    character: index
    for index, character in enumerate(vocab)
}

print("\nCHARACTER -> TOKEN ID")
print("---------------------")

for character, token_id in stoi.items():
    print(repr(character), "->", token_id)


# ============================================================
# 4. Token ID -> Character
# ============================================================

itos = {
    token_id: character
    for character, token_id in stoi.items()
}

print("\nTOKEN ID -> CHARACTER")
print("---------------------")

for token_id, character in itos.items():
    print(token_id, "->", repr(character))


# ============================================================
# 5. Encoder
# ============================================================

def encode(text: str) -> list[int]:
    """
    Convert text into token IDs.

    Example:
        "dog"
        -> [6, 16, 9]
    """

    return [stoi[character] for character in text]


# ============================================================
# 6. Decoder
# ============================================================

def decode(token_ids: list[int]) -> str:
    """
    Convert token IDs back into text.

    Example:
        [6, 16, 9]
        -> "dog"
    """

    return "".join(
        itos[token_id]
        for token_id in token_ids
    )


# ============================================================
# Test the tokenizer
# ============================================================

if __name__ == "__main__":

    sample_text = "dog"

    token_ids = encode(sample_text)

    print("\nTEST")
    print("----")

    print("Original text :", sample_text)
    print("Token IDs     :", token_ids)
    print("Decoded text  :", decode(token_ids))