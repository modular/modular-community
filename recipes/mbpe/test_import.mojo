"""Verify mbpe package is importable and Tokenizers.get works."""
from bpe.tokenizer import Tokenizers

def main() raises:
    var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
    print("gpt2:", gpt2.name())

    var cl100k = Tokenizers.get[Tokenizers.cl100k]()
    print("cl100k:", cl100k.name())

    var o200k = Tokenizers.get[Tokenizers.o200k]()
    print("o200k:", o200k.name())

    # Basic encode/decode roundtrip
    var text = "Hello, world!"
    var ids = gpt2.encode(text)
    var decoded = gpt2.decode(ids)
    print("roundtrip OK:", decoded == text)
