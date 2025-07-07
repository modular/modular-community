<!-- ![mojo_csv_logo](./mojo_csv_logo.png) -->
<image src='./mojo_csv_logo.png' width='900'/>

![language](https://img.shields.io/badge/language-mojo-orange)
![license](https://badgen.net/static/license/MIT/red)

# Mojo Csv

Csv parsing library written in pure Mojo

### usage

Add the Modular community channel (https://repo.prefix.dev/modular-community) to your pixi.toml file in the channels section.

```title:pixi.toml
channels = ["conda-forge", "https://conda.modular.com/max", "https://repo.prefix.dev/modular-community"]
```

`pixi add mojo_csv`


##### Basic Usage

```mojo
from mojo_csv import CsvReader
from pathlib import Path

fn main():
    var csv_path = Path("path/to/csv/file.csv")
    var reader = CsvReader(csv_path)
    for i in range(len(reader)):
        print(reader[i])
```

##### Optional Usage

```mojo
from mojo_csv import CsvReader
from pathlib import Path

fn main():
    var csv_path = Path("path/to/csv/file.csv")
    var reader = CsvReader(csv_path, delimiter="|", quotation_mark='*')
    for i in range(len(reader)):
        print(reader[i])
```

#### BETA
```mojo
ThreadedCsvReader(
    file_path: Path,
    delimiter: String = ",",
    quotation_mark: String = '"',
    num_threads: Int = 0 # 0 = use all available cores
)
```

### Example 1: Default (All Cores)

```mojo
var reader = ThreadedCsvReader(Path("large_file.csv"))
// Uses all 16 cores on a 16-core system
```

### Example 2: Custom Thread Count

```mojo
var reader = ThreadedCsvReader(Path("data.csv"), num_threads=4)
// Uses exactly 4 threads
```

### Example 3: Single-threaded

```mojo
var reader = ThreadedCsvReader(Path("data.csv"), num_threads=1)
// Forces single-threaded execution (same as CsvReader)
```

### Example 4: Custom Delimiter

````mojo
var reader = ThreadedCsvReader(
    Path("pipe_separated.csv"),
    delimiter="|",
    num_threads=8
)


### Attributes

```mojo
reader.raw : String # raw csv string
reader.raw_length : Int # total number of Chars
reader.headers : List[String] # first row of csv file
reader.row_count : Int  # total number of rows T->B
reader.column_count : Int # total number of columns L->R
reader.elements : List[String] # all delimited elements
reader.length : Int # total number of elements

```

##### Indexing

currently the array is only 1D, so indexing is fairly manual.

```Mojo
reader[0] # first element
```


### Performance

- average times over 1k iterations
- 7950x@5.8ghz (peak clock)
- uncompiled
- single-threaded

micro file benchmark (3 rows)
mini (100 rows)
small (1k rows)
medium file benchmark (100k rows)
large file benchmark (2m rows)

```log
✨ Pixi task (bench): mojo bench.mojo
running benchmark for micro csv:
average time in ms for micro file:
0.007699
-------------------------
running benchmark for mini csv:
average time in ms for mini file:
0.241136
-------------------------
running benchmark for small csv:
average time in ms for small file:
1.388513
-------------------------
running benchmark for medium csv:
average time in ms for medium file:
121.217188
-------------------------
running benchmark for large csv:
average time in ms for large file:
3582.876541
```

Performance comparison on various file sizes (average of multiple runs):

| File Size    | Single-threaded | Multi-threaded | Speedup |
| ------------ | --------------- | -------------- | ------- |
| 1,000 rows   | 1.42ms          | 1.30ms         | 1.09x   |
| 100,000 rows | 125ms           | 105ms          | 1.19x   |

_Tested on AMD 7950x (16 cores) @ 5.8GHz_

## Future Improvements

- [ ] SIMD optimization within each thread
- [ ] Async Chunking
- [ ] Streaming support for very large files
- [ ] Memory pool for reduced allocations
- [ ] Progress callbacks for long-running operation
