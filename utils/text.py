from typing import List


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    # Split long text into overlapping chunks to avoid cutting important context hard.
    if not text:
        return []
    if max_chars <= 0:
        return [text]

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks
