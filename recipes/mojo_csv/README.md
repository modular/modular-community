
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

```sh
pixi add mojo_csv
```

## Usage

By default uses all logical cores - 2
```mojo
 CsvReader(       
    in_csv: Path,
    delimiter: String = ",",
    quotation_mark: String = '"',
    num_threads: Int = 0, # default = 0 = use all available cores - 2
 )
```

```mojo
from mojo_csv import CsvReader
from pathlib import Path
from sys import exit

fn main() raises:
    var csv_path = Path("path/to/csv/file.csv")
    try:
        var reader = CsvReader(csv_path)
    except:
        exit()
    for i in range(len(reader)):
        print(reader[i])
```

### Delimiters

```mojo
    CsvReader(csv_path, delimiter=";", quotation_mark='|')
```

### Threads
__force single threaded__
```mojo
CsvReader(csv_pash, num_threads = 1)
```
__use all the threads__
```mojo
from sys import num_logical_cores

var reader = CsvReader(
    csv_path, num_threads = num_logical_cores()
)
```

### Attributes

```mojo
reader.raw : String # raw csv string
reader.raw_length : Int # total number of Chars
reader.headers : List[String] # first row of csv file
reader.row_count : Int  # total number of rows T->B
reader.column_count : Int # total number of columns L->R
reader.elements : List[String] # all delimited elements
reader.length : Int # total number of elements
````

### Indexing

currently the array is only 1D, so indexing is fairly manual.

```Mojo
reader[0] # first element
```

### Performance

- average times over 100-1k iterations
- AMD 7950x@5.8ghz
- single-threaded

micro file benchmark (3 rows) 
mini (100 rows) 
small (1k rows) 
medium file benchmark (100k rows) 
large file benchmark (2m rows) 

```log
✨ Pixi task (bench): mojo bench.mojo                                                                                                                                                      running benchmark for micro csv:
average time in ms for micro file:
0.0094 ms
-------------------------
running benchmark for mini csv:
average time in ms for mini file:
0.0657 ms
-------------------------
running benchmark for small csv:
average time in ms for small file:
0.317 ms
-------------------------
running benchmark for medium csv:
average time in ms for medium file:
24.62 ms
-------------------------
running benchmark for large csv:
average time in ms for large file:
878.6 ms
```

#### CSV Reader Performance Comparison
```
Small file benchmark (1,000 rows): 
Single-threaded: 
Average time: 0.455 ms 
Multi-threaded: 
Average time: 0.3744 ms 
Speedup: 1.22 x 

Medium file benchmark (100,000 rows): 
Single-threaded: 
Average time: 37.37 ms 
Multi-threaded: 
Average time: 24.46 ms 
Speedup: 1.53 x 

Large file benchmark (2,000,000 rows): 
Single-threaded: 
Average time: 1210.3 ms 
Multi-threaded: 
Average time: 863.9 ms 
Speedup: 1.4 x 

Summary:
Small file speedup: 1.22 x
Medium file speedup: 1.53 x
Large file speedup: 1.4 x
```

## Future Improvements

- [ ] 2D indexing
- [ ] CsvWriter
- [ ] CsvDictReader
- [ ] SIMD optimization within each thread
- [ ] Async Chunking
- [ ] Streaming support for very large files
- [ ] Memory pool for reduced allocations
- [ ] Progress callbacks for long-running operation
