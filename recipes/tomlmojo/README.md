# TOMLMojo <!-- omit in toc -->

A native TOML v1.0 parser for [Mojo](https://www.modular.com/mojo).

**[Repository on GitHub»](https://github.com/mojomath/decimojo/tree/main/src/tomlmojo)**　|　**[DeciMojo»](https://github.com/mojomath/decimojo)**

- [Overview](#overview)
- [History](#history)
- [Installation](#installation)
- [Quick start](#quick-start)
  - [Parse a TOML string](#parse-a-toml-string)
  - [Parse a TOML file](#parse-a-toml-file)
  - [Dotted keys and nested tables](#dotted-keys-and-nested-tables)
  - [Inline tables](#inline-tables)
  - [Arrays and arrays of tables](#arrays-and-arrays-of-tables)
  - [Integer formats and special floats](#integer-formats-and-special-floats)
- [Supported TOML features](#supported-toml-features)
- [API reference](#api-reference)
  - [Functions](#functions)
  - [`TOMLDocument`](#tomldocument)
  - [`TOMLValue`](#tomlvalue)
  - [`TOMLValueType`](#tomlvaluetype)
- [License](#license)

## Overview

TOMLMojo is a lightweight, pure-Mojo TOML parser (~1,500 LOC) that implements the core [TOML v1.0 specification](https://toml.io/en/v1.0.0). It parses TOML source text into a `TOMLDocument` — a nested dictionary structure that you can query by key, table name, or array index. It handles all common TOML constructs including nested tables, inline tables, dotted keys, arrays of tables, and all TOML data types.

## History

TOMLMojo was initially developed in **April 2025** alongside [DeciMojo](https://github.com/mojomath/decimojo) v0.3.0 to support DeciMojo's TOML-based test data system. At that time, no TOML parser existed in the Mojo ecosystem, so I wrote one from scratch.

Currently, **DeciMojo heavily depends on TOMLMojo** for its entire testing and benchmarking infrastructure. All test cases and benchmark configurations are stored as TOML files and loaded via TOMLMojo's `parse_file()` API.

While originally built for internal use, TOMLMojo has grown into a general-purpose TOML library suitable for any Mojo project that needs to parse configuration files or structured data.

The version number of TOMLMojo is aligned with the latest version of DeciMojo.

## Installation

TOMLMojo is available in the modular-community `https://repo.prefix.dev/modular-community` package repository. Add it to your `channels` in `pixi.toml`:

```toml
channels = ["https://conda.modular.com/max", "https://repo.prefix.dev/modular-community", "conda-forge"]
```

Then install:

```bash
pixi add tomlmojo
```

## Quick start

### Parse a TOML string

```mojo
import tomlmojo

fn main() raises:
    var doc = tomlmojo.parse_string("""
        title = "My App"
        version = 42
        debug = true

        [server]
        host = "localhost"
        port = 8080

        [database]
        name = "mydb"
        max_connections = 100
    """)

    # Access top-level values
    print(doc.get("title").as_string())     # My App
    print(doc.get("version").as_int())      # 42
    print(doc.get("debug").as_bool())       # True

    # Access table values
    var server = doc.get_table("server")
    print(server["host"].as_string())       # localhost
    print(server["port"].as_int())          # 8080
```

### Parse a TOML file

```mojo
import tomlmojo

fn main() raises:
    var doc = tomlmojo.parse_file("config.toml")
    var db = doc.get_table("database")
    print(db["name"].as_string())           # mydb
```

### Dotted keys and nested tables

```mojo
import tomlmojo

fn main() raises:
    var doc = tomlmojo.parse_string("""
        fruit.name = "apple"
        fruit.color = "red"
        fruit.size.width = 10
        fruit.size.height = 20

        [server]
        host = "localhost"

        [server.database]
        name = "mydb"
        port = 5432
    """)

    # Dotted keys create nested tables automatically
    var fruit = doc.get("fruit").as_table()
    print(fruit["name"].as_string())        # apple
    var size = fruit["size"].as_table()
    print(size["width"].as_int())           # 10

    # Dotted table headers also nest
    var server = doc.get("server").as_table()
    var db = server["database"].as_table()
    print(db["name"].as_string())           # mydb
```

### Inline tables

```mojo
import tomlmojo

fn main() raises:
    var doc = tomlmojo.parse_string("""
        point = {x = 1, y = 2}
        animal = {type.name = "pug"}
    """)

    var pt = doc.get("point").as_table()
    print(pt["x"].as_int())                 # 1
    print(pt["y"].as_int())                 # 2

    # Dotted keys work inside inline tables too
    var animal = doc.get("animal").as_table()
    var typ = animal["type"].as_table()
    print(typ["name"].as_string())          # pug
```

### Arrays and arrays of tables

```mojo
import tomlmojo

fn main() raises:
    var doc = tomlmojo.parse_string("""
        colors = [
            "red",
            "green",
            "blue",
        ]

        [[servers]]
        name = "alpha"
        ip = "10.0.0.1"

        [[servers]]
        name = "beta"
        ip = "10.0.0.2"
    """)

    # Simple arrays
    var colors = doc.get("colors").as_array()
    print(len(colors))                      # 3
    print(colors[0].as_string())            # red

    # Array of tables
    var servers = doc.get_array_of_tables("servers")
    print(len(servers))                     # 2
    print(servers[0]["name"].as_string())   # alpha
    print(servers[1]["ip"].as_string())     # 10.0.0.2
```

### Integer formats and special floats

```mojo
import tomlmojo

fn main() raises:
    var doc = tomlmojo.parse_string("""
        hex = 0xDEADBEEF
        oct = 0o755
        bin = 0b11010110
        dec = 1_000_000
        pos_inf = inf
        neg_inf = -inf
        not_a_num = nan
    """)

    print(doc.get("hex").as_int())          # 3735928559
    print(doc.get("oct").as_int())          # 493
    print(doc.get("bin").as_int())          # 214
    print(doc.get("dec").as_int())          # 1000000
```

## Supported TOML features

| Feature                   | Example                      | Supported |
| ------------------------- | ---------------------------- | --------- |
| Basic strings             | `"hello\nworld"`             | Yes       |
| Literal strings           | `'no\escape'`                | Yes       |
| Multiline basic strings   | `"""..."""`                  | Yes       |
| Multiline literal strings | `'''...'''`                  | Yes       |
| Unicode escapes           | `"\u0041"` → `A`             | Yes       |
| Integers (decimal)        | `42`, `1_000`                | Yes       |
| Integers (hex/oct/bin)    | `0xDEAD`, `0o755`, `0b1101`  | Yes       |
| Floats                    | `3.14`, `1e10`, `inf`, `nan` | Yes       |
| Signed numbers            | `+42`, `-3.14`, `+inf`       | Yes       |
| Booleans                  | `true`, `false`              | Yes       |
| Arrays                    | `[1, 2, 3]`                  | Yes       |
| Multiline arrays          | Trailing commas, newlines    | Yes       |
| Standard tables           | `[table]`                    | Yes       |
| Dotted table headers      | `[a.b.c]`                    | Yes       |
| Inline tables             | `{x = 1, y = 2}`             | Yes       |
| Dotted keys               | `a.b.c = "val"`              | Yes       |
| Quoted keys               | `"my key" = "val"`           | Yes       |
| Array of tables           | `[[items]]`                  | Yes       |
| Duplicate key detection   | Error on redefinition        | Yes       |
| Comments                  | `# comment`                  | Yes       |
| Datetime types            | `1979-05-27T07:32:00`        | No        |

## API reference

### Functions

| Function                                      | Description         |
| --------------------------------------------- | ------------------- |
| `parse_string(input: String) -> TOMLDocument` | Parse a TOML string |
| `parse_file(path: String) -> TOMLDocument`    | Parse a TOML file   |

### `TOMLDocument`

| Method                     | Returns                         | Description                 |
| -------------------------- | ------------------------------- | --------------------------- |
| `get(key)`                 | `TOMLValue`                     | Get a top-level value       |
| `get_table(name)`          | `Dict[String, TOMLValue]`       | Get a table as a dictionary |
| `get_array(key)`           | `List[TOMLValue]`               | Get an array                |
| `get_array_of_tables(key)` | `List[Dict[String, TOMLValue]]` | Get an array of tables      |

### `TOMLValue`

| Method        | Returns                   | Description                |
| ------------- | ------------------------- | -------------------------- |
| `as_string()` | `String`                  | Get value as string        |
| `as_int()`    | `Int`                     | Get value as integer       |
| `as_float()`  | `Float64`                 | Get value as float         |
| `as_bool()`   | `Bool`                    | Get value as boolean       |
| `as_table()`  | `Dict[String, TOMLValue]` | Get value as table dict    |
| `as_array()`  | `List[TOMLValue]`         | Get value as array         |
| `is_table()`  | `Bool`                    | Check if value is a table  |
| `is_array()`  | `Bool`                    | Check if value is an array |

### `TOMLValueType`

Constants: `NULL`, `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `ARRAY`, `TABLE`.

## License

TOMLMojo is part of the [DeciMojo](https://github.com/mojomath/decimojo) project and is licensed under the Apache License v2.0.
