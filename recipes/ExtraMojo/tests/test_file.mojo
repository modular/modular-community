from collections import Dict
from pathlib import Path
from python import Python
from tensor import Tensor
from testing import *

from ExtraMojo.bstr.bstr import SplitIterator
from ExtraMojo.io.delimited import (
    DelimReader,
    FromDelimited,
    ToDelimited,
    DelimWriter,
)
from ExtraMojo.io.buffered import (
    BufferedReader,
    read_lines,
    for_each_line,
    BufferedWriter,
)


fn s(bytes: Span[UInt8]) -> String:
    """Convert bytes to a String."""
    var buffer = String()
    buffer.write_bytes(bytes)
    return buffer


fn strings_for_writing(size: Int) -> List[String]:
    var result = List[String]()
    for i in range(size):
        result.append(
            "Line: " + String(i) + " X" + ("-" * 64)
        )  # make lines long
    return result


fn test_read_until(file: Path, expected_lines: List[String]) raises:
    var buffer_capacities = List(10, 100, 200, 500)
    for cap in buffer_capacities:
        var fh = open(file, "r")
        var reader = BufferedReader(fh^, buffer_capacity=cap[])
        var buffer = List[UInt8]()
        var counter = 0
        while reader.read_until(buffer) != 0:
            assert_equal(List(expected_lines[counter].as_bytes()), buffer)
            counter += 1
            buffer.clear()

        assert_equal(counter, len(expected_lines))
        print(
            String("Successful read_until with buffer capacity of {}").format(
                cap[]
            )
        )


fn test_read_until_return_trailing(
    file: Path, expected_lines: List[String]
) raises:
    var fh = open(file, "r")
    var reader = BufferedReader(fh^, buffer_capacity=200)
    var buffer = List[UInt8]()
    var counter = 0
    while reader.read_until(buffer) != 0:
        assert_equal(List(expected_lines[counter].as_bytes()), buffer)
        counter += 1
        buffer.clear()
    assert_equal(counter, len(expected_lines))
    print("Successful read_until_return_trailing")


fn test_read_bytes(file: Path) raises:
    var fh = open(file, "r")
    var reader = BufferedReader(fh^, buffer_capacity=50)
    var buffer = List[UInt8](capacity=125)
    for _ in range(0, 125):
        buffer.append(0)
    var found_file = List[UInt8]()

    # Read bytes from the buf reader, copy to found
    var bytes_read = 0
    while True:
        bytes_read = reader.read_bytes(buffer)
        if bytes_read == 0:
            break
        found_file.extend(buffer[0:bytes_read])
    # Last usage of reader, meaning it should call __del__ here.

    var expected = open(file, "r").read().as_bytes()
    assert_equal(len(expected), len(found_file))
    for i in range(0, len(expected)):
        assert_equal(
            expected[i], found_file[i], msg="Unequal at byte: " + String(i)
        )
    print("Successful read_bytes")


fn test_context_manager_simple(file: Path, expected_lines: List[String]) raises:
    var buffer = List[UInt8]()
    var counter = 0
    with BufferedReader(open(file, "r"), buffer_capacity=200) as reader:
        while reader.read_until(buffer) != 0:
            assert_equal(List(expected_lines[counter].as_bytes()), buffer)
            counter += 1
            buffer.clear()
    assert_equal(counter, len(expected_lines))
    print("Successful read_until")


fn test_read_lines(file: Path, expected_lines: List[String]) raises:
    var lines = read_lines(String(file))
    assert_equal(len(lines), len(expected_lines))
    for i in range(0, len(lines)):
        assert_equal(lines[i], List(expected_lines[i].as_bytes()))
    print("Successful read_lines")


fn test_for_each_line(file: Path, expected_lines: List[String]) raises:
    var counter = 0
    var found_bad = False

    @parameter
    fn inner(buffer: Span[UInt8], start: Int, end: Int) capturing -> None:
        if s(buffer[start:end]) != expected_lines[counter]:
            found_bad = True
        counter += 1

    for_each_line[inner](String(file))
    assert_false(found_bad)
    print("Successful for_each_line")


@value
struct SerDerStruct(ToDelimited, FromDelimited):
    var index: Int
    var name: String

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        writer.write_record(self.index, self.name)

    fn write_header(read self, mut writer: DelimWriter) raises:
        writer.write_record("index", "name")

    @staticmethod
    fn from_delimited(
        mut data: SplitIterator,
        read header_values: Optional[List[String]] = None,
    ) raises -> Self:
        var index = Int(StringSlice(unsafe_from_utf8=data.__next__()))
        var name = String()  # String constructor expected nul terminated byte span
        name.write_bytes(data.__next__())
        return Self(index, name)


fn test_delim_reader_writer(file: Path) raises:
    var to_write = List[SerDerStruct]()
    for i in range(0, 1000):
        to_write.append(SerDerStruct(i, String("MyNameIs" + String(i))))
    var writer = DelimWriter(
        BufferedWriter(open(String(file), "w")), delim="\t", write_header=True
    )
    for item in to_write:
        writer.serialize(item[])
    writer.flush()
    writer.close()

    var reader = DelimReader[SerDerStruct](
        BufferedReader(open(String(file), "r")),
        delim=ord("\t"),
        has_header=True,
    )
    var count = 0
    for item in reader^:
        assert_equal(to_write[count].index, item.index)
        assert_equal(to_write[count].name, item.name)
        count += 1
    assert_equal(count, len(to_write))
    print("Successful delim_writer")


@value
struct ThinWrapper(ToDelimited, FromDelimited):
    var stuff: Dict[String, Int]

    fn write_to_delimited(read self, mut writer: DelimWriter) raises:
        var seen = 1
        for value in self.stuff.values():  # Relying on stable iteration order
            writer.write_field(value[], is_last=seen == len(self.stuff))
            seen += 1

    fn write_header(read self, mut writer: DelimWriter) raises:
        var seen = 1
        for header in self.stuff.keys():  # Relying on stable iteration order
            writer.write_field(header[], is_last=seen == len(self.stuff))
            seen += 1

    @staticmethod
    fn from_delimited(
        mut data: SplitIterator,
        read header_values: Optional[List[String]] = None,
    ) raises -> Self:
        var result = Dict[String, Int]()
        for header in header_values.value():
            result[header[]] = Int(
                StringSlice(unsafe_from_utf8=data.__next__())
            )
        return Self(result)


fn test_delim_reader_writer_dicts(file: Path) raises:
    var to_write = List[ThinWrapper]()
    var headers = List(
        String("a"), String("b"), String("c"), String("d"), String("e")
    )
    for i in range(0, 1000):
        var stuff = Dict[String, Int]()
        for header in headers:
            stuff[header[]] = i
        to_write.append(ThinWrapper(stuff))
    var writer = DelimWriter(
        BufferedWriter(open(String(file), "w")),
        delim="\t",
        write_header=True,
    )
    for item in to_write:
        writer.serialize(item[])
    writer.flush()
    writer.close()

    var reader = DelimReader[ThinWrapper](
        BufferedReader(open(String(file), "r")),
        delim=ord("\t"),
        has_header=True,
    )
    var count = 0
    for item in reader^:
        for header in headers:
            assert_equal(to_write[count].stuff[header[]], item.stuff[header[]])
        count += 1
    assert_equal(count, len(to_write))
    print("Successful delim_writer")


fn test_buffered_writer(file: Path, expected_lines: List[String]) raises:
    var fh = BufferedWriter(open(String(file), "w"), buffer_capacity=128)
    for i in range(len(expected_lines)):
        fh.write_bytes(expected_lines[i].as_bytes())
        fh.write_bytes("\n".as_bytes())
    fh.flush()
    fh.close()

    test_read_until(String(file), expected_lines)


fn create_file(path: String, lines: List[String]) raises:
    with open(path, "w") as fh:
        for i in range(len(lines)):
            fh.write(lines[i])
            fh.write(String("\n"))


fn create_file_no_trailing_newline(path: String, lines: List[String]) raises:
    with open(path, "w") as fh:
        for i in range(len(lines)):
            fh.write(lines[i])
            if i != len(lines) - 1:
                fh.write(String("\n"))


fn main() raises:
    var tempfile = Python.import_module("tempfile")
    var tempdir = tempfile.TemporaryDirectory()
    var file = Path(String(tempdir.name)) / "lines.txt"
    var file_no_trailing_newline = Path(
        String(tempdir.name)
    ) / "lines_no_trailing_newline.txt"
    var strings = strings_for_writing(10000)
    create_file(String(file), strings)
    create_file_no_trailing_newline(String(file_no_trailing_newline), strings)

    # Tests
    test_read_until(String(file), strings)
    test_read_until_return_trailing(String(file_no_trailing_newline), strings)
    test_read_bytes(String(file))
    test_read_lines(String(file), strings)
    test_for_each_line(String(file), strings)
    var buf_writer_file = Path(String(tempdir.name)) / "buf_writer.txt"
    test_buffered_writer(String(buf_writer_file), strings)
    var delim_file = Path(String(tempdir.name)) / "delim.txt"
    test_delim_reader_writer(String(delim_file))
    var delim_dict_file = Path(String(tempdir.name)) / "delim_dict.txt"
    test_delim_reader_writer_dicts(String(delim_dict_file))
    print("SUCCESS")

    _ = tempdir.cleanup()
