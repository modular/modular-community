# ArgMojo

![icon](image.jpeg)

A command-line argument parser library for [Mojo](https://www.modular.com/mojo).

> **A**rguments  
> **R**esolved and  
> **G**rouped into  
> **M**eaningful  
> **O**ptions and  
> **J**oined  
> **O**bjects

## Overview

ArgMojo provides a builder-pattern API for defining and parsing command-line arguments in Mojo. It currently supports:

- **Long options**: `--verbose`, `--output file.txt`, `--output=file.txt`
- **Short options**: `-v`, `-o file.txt`
- **Boolean flags**: options that take no value
- **Positional arguments**: matched by position
- **Default values**: fallback when an argument is not provided
- **Required arguments**: validation that mandatory args are present
- **Auto-generated help**: `--help` / `-h` (no need to implement manually)
- **Version display**: `--version` / `-V` (also auto-generated)
- **`--` stop marker**: everything after `--` is treated as positional
- **Short flag merging**: `-abc` expands to `-a -b -c`
- **Attached short values**: `-ofile.txt` means `-o file.txt`
- **Choices validation**: restrict values to a set (e.g., `json`, `csv`, `table`)
- **Metavar**: custom display name for values in help text
- **Hidden arguments**: exclude internal args from `--help` output
- **Count flags**: `-vvv` → `get_count("verbose") == 3`
- **Positional arg count validation**: reject extra positional args
- **Negatable flags**: `--color` / `--no-color` paired flags with `.negatable()`
- **Mutually exclusive groups**: prevent conflicting flags (e.g., `--json` vs `--yaml`)
- **Required-together groups**: enforce that related flags are provided together (e.g., `--username` + `--password`)
- **Long option prefix matching**: allow abbreviated options (e.g., `--verb` → `--verbose`). If the prefix is ambiguous (e.g., `--ver` could match both `--verbose` and `--version`), an error is raised.

---

I created this project to support my experiments with a CLI-based Chinese character search engine in Mojo, as well as a CLI-based calculator for [DeciMojo](https://github.com/forfudan/decimojo). It is inspired by Python's `argparse`, Rust's `clap`, Go's `cobra`, and other popular argument parsing libraries, but designed to fit Mojo's unique features and constraints.

My goal is to provide a Mojo-idiomatic argument parsing library that can be easily adopted by the growing Mojo community for their CLI applications. **Before Mojo v1.0** (which means it gets stable), my focus is on building core features and ensuring correctness. "Core features" refer to those who appear in `argparse`/`clap`/`cobra` and are commonly used in CLI apps. "Correctness" means that the library should handle edge cases properly, provide clear error messages, and have good test coverage. Some fancy features will depend on my time and interest.

## Installation

ArgMojo requires Mojo == 0.26.1 and uses [pixi](https://pixi.sh) for environment management.

```bash
git clone https://github.com/forfudan/argmojo.git
cd argmojo
pixi install
```

I make the Mojo version strictly 0.26.1 because that's the version I developed and tested on, and Mojo is rapidly evolving. Based on my experience, the library will not work every time there's a new Mojo release.

## Quick Start

Here is a simple example of how to use ArgMojo in a Mojo program. The full example can be found in `examples/demo.mojo`.

```mojo
from argmojo import Arg, Command


fn main() raises:
    var cmd = Command("demo", "A CJK-aware text search tool that supports Pinyin and Yuhao Input Methods (宇浩系列輸入法).", version="0.1.0")

    # Positional arguments
    cmd.add_arg(Arg("pattern", help="Search pattern").positional().required())
    cmd.add_arg(Arg("path", help="Search path").positional().default("."))

    # Boolean flags
    cmd.add_arg(
        Arg("ling", help="Use Lingming IME for encoding")
        .long("ling").short("l").flag()
    )

    # Count flag (verbosity)
    cmd.add_arg(
        Arg("verbose", help="Increase verbosity (-v, -vv, -vvv)")
        .long("verbose").short("v").count()
    )

    # Key-value option with choices and metavar
    var formats: List[String] = ["json", "csv", "table"]
    cmd.add_arg(
        Arg("format", help="Output format")
        .long("format").short("f").choices(formats^).default("table")
    )

    # Negatable flag — --color enables, --no-color disables
    cmd.add_arg(
        Arg("color", help="Enable colored output")
        .long("color").flag().negatable()
    )

    # Parse and use
    var result = cmd.parse()
    print("pattern:", result.get_string("pattern"))
    print("verbose:", result.get_count("verbose"))
    print("format: ", result.get_string("format"))
    print("color:  ", result.get_flag("color"))
```

## Usage Examples

For detailed explanations and more examples of every feature, see the **[User Manual](docs/user_manual.md)**.

Build the demo binary first, then try the examples below:

```bash
pixi run build_demo
```

### Basic usage — positional args and flags

```bash
./demo "nihao" ./src --ling
```

### Short options and default values

The second positional arg (`path`) defaults to `"."` when omitted:

```bash
./demo "zhongguo" -l
```

### Help and version

```bash
./demo --help
./demo --version
```

### Merged short flags

Multiple short flags can be combined in a single `-` token. For example, `-liv` is equivalent to `-l -i -v`:

```bash
./demo "pattern" ./src -liv
```

### Attached short values

A short option can receive its value without a space:

```bash
./demo "pattern" -d3          # same as -d 3
./demo "pattern" -ftable      # same as -f table
```

### Count flags — verbosity

Use `-v` multiple times (merged or repeated) to increase verbosity:

```bash
./demo "pattern" -v           # verbose = 1
./demo "pattern" -vvv         # verbose = 3
./demo "pattern" -v --verbose # verbose = 2  (short + long)
```

### Choices validation

The `--format` option only accepts `json`, `csv`, or `table`:

```bash
./demo "pattern" --format json     # OK
./demo "pattern" --format xml      # Error: Invalid value 'xml' for 'format'. Valid choices: json, csv, table
```

### Negatable flags

A negatable flag pairs `--X` (sets `True`) with `--no-X` (sets `False`) automatically:

```bash
./demo "pattern" --color           # color = True
./demo "pattern" --no-color        # color = False
./demo "pattern"                   # color = False (default)
```

### Mutually exclusive groups

`--json` and `--yaml` are mutually exclusive — using both is an error:

```bash
./demo "pattern" --json            # OK
./demo "pattern" --yaml            # OK
./demo "pattern" --json --yaml     # Error: Arguments are mutually exclusive: '--json', '--yaml'
```

### `--` stop marker

Everything after `--` is treated as a positional argument, even if it starts with `-`:

```bash
./demo --ling -- "--pattern-with-dashes" ./src
```

### Hidden arguments

Some arguments are excluded from `--help` but still work at the command line (useful for debug flags):

```bash
./demo "pattern" --debug-index   # Works, but not shown in --help
```

### Required-together groups

`--username` and `--password` must be provided together — using one without the other is an error:

```bash
./demo "pattern" --username admin --password secret  # OK
./demo "pattern"                                     # OK (neither is provided)
./demo "pattern" --username admin                    # Error: '--password' required when '--username' is provided
```

### A mock example showing how features work together

```bash
./demo yes ./src --verbo --color -li -d 3 --no-color --usern zhu --pas 12345
```

This will be parsed as:

```bash
=== Parsed Arguments ===
  pattern: yes
  path: ./src
  -l, --ling            True
  -i, --ignore-case     True
  -v, --verbose         1
  -d, --max-depth       3
  -f, --format          table
  --color               False
  --json                False
  --yaml                False
  -u, --username        zhu
  -p, --password        12345
```

## Development

```bash
# Format code
pixi run format

# Build package
pixi run package

# Run tests
pixi run test

# Clean build artifacts
pixi run clean
```

## Project Structure

```txt
argmojo/
├── docs/                           # Documentation
│   ├── user_manual.md              # User manual with detailed examples
│   └── argmojo_overall_planning.md
├── src/
│   └── argmojo/                    # Main package
│       ├── __init__.mojo           # Package exports
│       ├── arg.mojo                # Arg struct (argument definition)
│       ├── command.mojo            # Command struct (parsing logic)
│       └── result.mojo             # ParseResult struct (parsed values)
├── tests/
│   └── test_argmojo.mojo           # Tests
├── pixi.toml                       # pixi configuration
├── .gitignore
├── LICENSE
└── README.md
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
