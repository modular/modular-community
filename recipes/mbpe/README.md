# mbpe

A high-performance, trainable BPE tokenizer written in **Mojo**, compatible with OpenAI's `tiktoken` API and `.tiktoken` vocabulary format.

## Features

- **tiktoken-compatible API** — same `get_encoding()`, `encode()`/`decode()` semantics, `allowed_special`/`disallowed_special` handling, and `.tiktoken` vocabulary files. Point existing code at `mbpe` and it works.
- **Fast** — a Mojo-native core that beats `tiktoken-rs` (Rust) on encode and decode across the GPT-2, GPT-4 and GPT-4o encodings.
- **Fast Python bindings** — substantially faster than Python `tiktoken`, while staying competitive with `tiktoken-rs`.
- **Train your own** — `tokenizer.train(["hello world"], vocab_size=300)` and save directly to `.tiktoken` format.
- **Extensible by design** — the `PreTokenizer` is a Mojo trait, not a hardcoded implementation. Ships with GPT-2, GPT-4 (cl100k) and GPT-4o (o200k) pipelines; write your own to match your data.

## Usage

### Mojo

```mojo
from bpe import Tokenizers

var tokenizer = Tokenizers.get[Tokenizers.gpt2]()
```

### Python

```python
import mbpe

tokenizer = mbpe.get_encoding("gpt2")
tokens = tokenizer.encode("hello world")
print(tokens)                        # [31373, 995]
print(tokenizer.decode(tokens))      # "hello world"
```

## Installation

Add the modular-community channel and install with pixi:

```toml
[workspace]
channels = ["https://repo.prefix.dev/modular-community"]
```

```bash
pixi add mbpe
```

The Python bindings are also published to PyPI as `mbpe` (`pip install mbpe`).

## License

[MIT](https://github.com/ratulb/mbpe/blob/main/LICENSE)
