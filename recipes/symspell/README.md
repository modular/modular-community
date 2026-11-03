# symspell.mojo

Fast spelling correction in pure Mojo. It is a port of
[symspellpy](https://github.com/mammothb/symspellpy) and of Wolf Garbe's
[SymSpell](https://github.com/wolfgarbe/SymSpell), and it gives exactly the same results as
symspellpy 6.10.0.

- About 10× faster than symspellpy for lookups and for building a dictionary.
- Uses less memory. A saved index opens in milliseconds.
- Read-only indexes, so many threads can share one. A parallel batch lookup is included.
- Tested against symspellpy on about 190,000 lookups, sentences and phrases, in English and other
  languages.

## What SymSpell does

SymSpell finds the dictionary words that are close to a misspelled word. "Close" means a small
edit distance: the number of single-letter insertions, deletions, substitutions and swaps needed
to turn one word into the other. "helo" is 1 edit away from "hello", "help" and "held".

Comparing a word with every dictionary word is slow. SymSpell precomputes, for every dictionary
word, all the strings you get by deleting up to N letters. At lookup time it deletes letters from
the input and looks those strings up in that table. This is why lookups are fast, and why building
a dictionary takes a moment.

On top of single-word lookups, it can:

- correct a whole sentence, including wrongly split or joined words (`lookup_compound`);
- split text without spaces into words (`word_segmentation`).

## Install

With [pixi](https://pixi.sh), once the package is on the modular-community channel:

```toml
[workspace]
channels = ["https://conda.modular.com/max", "https://repo.prefix.dev/modular-community", "conda-forge"]

[dependencies]
mojo = "1.2.*"
symspell = "*"
```

From source: clone this repository and run `pixi run build`. This writes
`build/symspell.mojoc`. Compile your program with `mojo run -I build your_program.mojo`.

Requires Mojo 1.2 on Linux (x86-64 or arm64) or macOS (Apple silicon).

## Quick start

The English dictionaries in `data/` come from symspellpy.

```mojo
from symspell import SymSpellBuilder, Verbosity


def main() raises:
    var builder = SymSpellBuilder()  # max edit distance 2, prefix length 7
    builder.load_dictionary("data/frequency_dictionary_en_82_765.txt")
    builder.load_bigram_dictionary("data/frequency_bigramdictionary_en_243_342.txt")
    var index = builder^.build()

    for suggestion in index.lookup("memebers", Verbosity.CLOSEST):
        print(suggestion.term, suggestion.distance, suggestion.count)
    # members 1 226656153

    print(index.lookup_compound("whereis th elove", max_edit_distance=2)[0].term)
    # where is the love

    print(index.word_segmentation("thequickbrownfox").corrected_string)
    # the quick brown fox
```

Build once, save, and open the saved file later. Opening takes milliseconds:

```mojo
index.save("en.idx")
var index = SymSpell.open("en.idx")
```

More in [`examples/`](examples): saving and opening, reusing a scratch buffer, and batch lookups.
Run them all with `pixi run examples`.

## Command-line tool

`pixi run cli` builds `build/bin/symspell`:

```console
$ symspell build data/frequency_dictionary_en_82_765.txt -o en.idx
$ symspell check notes.txt --index en.idx
notes.txt:1:1: Ths -> The
notes.txt:2:21: skiped -> skipped
$ symspell lookup helo --index en.idx
$ symspell correct "whereis th elove" --index en.idx
$ symspell segment thequickbrownfox --index en.idx
```

`check` exits with status 1 if it finds unknown words. Without `--index`, it builds the English
dictionary from `data/` first.

## API overview

| | |
|---|---|
| `SymSpellConfig(max_dictionary_edit_distance=2, prefix_length=7, count_threshold=1, distance_algorithm=DAMERAU_OSA)` | Settings fixed when the index is built. |
| `SymSpellBuilder(config)` | A mutable dictionary. |
| `.load_dictionary(path)`, `.load_bigram_dictionary(path)`, `.create_dictionary(corpus_path)` | Load words from files. |
| `.create_dictionary_entry(word, count)`, `.delete_dictionary_entry(word)` | Edit single words. |
| `.build(threads=0)`, `.snapshot(threads=0)` | Make a read-only `SymSpell` index. |
| `SymSpell.open(path)`, `index.save(path)` | Load and save an index. |
| `index.lookup(word, verbosity, max_edit_distance, include_unknown, ignore_token, transfer_casing)` | Suggestions for one word. |
| `index.lookup_batch(words, verbosity, ..., threads=0)` | Many words, in parallel. |
| `index.lookup_compound(text, max_edit_distance, ...)` | Correct a sentence. |
| `index.word_segmentation(text, max_edit_distance, ...)` | Insert missing spaces. |
| `index.contains(word)`, `index.count(word)`, `index.word_count()`, ... | Inspect the dictionary. |

`Verbosity.TOP` returns the best suggestion. `CLOSEST` returns all suggestions at the smallest
distance. `ALL` returns everything within the maximum distance. Suggestions are sorted by distance,
then by count.

Each public function has a docstring. [docs/BEHAVIOR.md](docs/BEHAVIOR.md) describes the exact
behavior, rule by rule.

## Differences from symspellpy

The names and results are the same. The main differences:

- A mutable `SymSpellBuilder` makes a read-only `SymSpell` index. Adding words later needs a rebuild.
- Errors raise instead of returning `False`.
- `ignore_token` takes a function instead of a regular expression.
- A binary index file replaces pickle.
- `word_segmentation` does not apply Unicode NFKC normalization.

The full list is in [docs/API_DIFFERENCES.md](docs/API_DIFFERENCES.md).

## Speed and memory

Measured against symspellpy 6.10.0 with editdistpy, on the English dictionary, on a 4-thread Intel
laptop CPU. Details and more numbers are in [BENCHMARKS.md](BENCHMARKS.md).

| | symspellpy | symspell | |
|---|---|---|---|
| Build the dictionary | 1.87 s | 0.17 s | 10.8× |
| Load a saved index | 0.90 s | 0.05 s | 20× |
| `lookup`, `TOP`, distance 2 | 40 µs | 3.8 µs | 10.5× |
| `lookup`, `CLOSEST`, distance 2 | 41 µs | 4.2 µs | 9.8× |
| `lookup_compound`, per sentence | 3.3 ms | 0.30 ms | 11× |
| Memory with a loaded index | 152 MB | 26 MB | 5.9× less |

## Limitations

- No Unicode normalization. Normalize the input yourself if it may contain ligatures or other
  characters that NFKC would change.
- Linux and macOS only, on little-endian machines. An index file is limited to 4 GiB.
- Dictionary files must be UTF-8.
- A saved index is larger than symspellpy's pickle (24 MB against 5.3 MB for English), because it is
  laid out to be used without parsing.

## Development

| Command | What it does |
|---|---|
| `pixi run test` | All tests. The first run generates large reference files with symspellpy (about 2 minutes). |
| `pixi run test-debug` | All tests with debug assertions. |
| `pixi run examples` | Runs the examples and checks the command-line tool. |
| `pixi run bench` | Benchmarks against symspellpy. `BENCH_RUNS=3` takes medians of 3 runs. |
| `pixi run ref` | Regenerates every reference file with symspellpy. |
| `pixi run fmt` | Formats the Mojo code. |

Design notes are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Credits

- [SymSpell](https://github.com/wolfgarbe/SymSpell) by Wolf Garbe, the original algorithm and C#
  implementation (MIT).
- [symspellpy](https://github.com/mammothb/symspellpy) by mmb L, the Python port whose behavior,
  tests and dictionaries this library uses (MIT).

symspell is released under the MIT License. See [LICENSE](LICENSE), which includes both projects'
notices.
