from std.pathlib import Path, cwd
from std.testing import assert_true

from mojo_csv import CsvReader


fn test_basic_lf() raises:
    """Test basic comma-separated parsing with LF line endings."""
    var expected = List[String]()
    expected.append("item1")
    expected.append("item2")
    expected.append('"ite,em3"')
    expected.append('"p""ic"')
    expected.append(" pi c")
    expected.append("pic")
    expected.append("r_i_1")
    expected.append('"r_i_2"""')
    expected.append("r_i_3")

    var in_csv: Path = cwd().joinpath("test_data_lf.csv")
    in_csv.write_text(
        'item1,item2,"ite,em3"\n'
        '"p""ic", pi c,pic,\n'
        'r_i_1,"r_i_2""",r_i_3,\n'
    )
    var rd = CsvReader(in_csv)
    assert_true(rd.col_count == 3)
    for x in range(len(rd)):
        assert_true(
            rd.elements[x] == expected[x],
            String("[{0}] != expected [{1}] at index {2}").format(
                rd.elements[x], expected[x], x
            ),
        )
    assert_true(rd.row_count == 3)
    assert_true(len(rd.elements) == 9)
    print("✅ test_basic_lf passed")


fn test_crlf_single_threaded() raises:
    """Test CRLF line endings parsed single-threaded.

    This catches the bug where ord(\"\\r\\n\") was incorrectly used
    (ord takes a single char) and where col_start after \\r pointed
    at the \\n byte, including it in the next element string.
    """
    var expected = List[String]()
    expected.append("a")
    expected.append("b")
    expected.append("c")
    expected.append("d")
    expected.append("e")
    expected.append("f")

    var in_csv: Path = cwd().joinpath("test_data_crlf.csv")
    in_csv.write_text("a,b,c\r\nd,e,f\r\n")

    var rd = CsvReader(in_csv, num_threads=1)
    assert_true(rd.col_count == 3, "crlf col_count mismatch")
    assert_true(rd.row_count == 2, "crlf row_count mismatch")
    assert_true(len(rd) == 6, "crlf element count mismatch")
    for i in range(len(expected)):
        assert_true(
            rd[i] == expected[i],
            String("crlf [{0}] != expected [{1}] at index {2}").format(
                rd[i], expected[i], i
            ),
        )
    print("✅ test_crlf_single_threaded passed")


fn test_threaded_vs_single_threaded() raises:
    """Test that threaded and single-threaded parsers produce identical results.

    This catches the bug where single-threaded CRLF handling included
    the \\n byte in the next element, causing mismatches with threaded output.
    """
    # Build a CSV large enough to trigger multi-threaded parsing (>1000 bytes)
    var csv_content = String("header1,header2,header3\n")
    for i in range(50):
        csv_content += String("val{0},val{0},val{0}\n").format(i)

    var in_csv: Path = cwd().joinpath("test_data_threaded.csv")
    in_csv.write_text(csv_content)

    var single = CsvReader(in_csv, num_threads=1)
    var threaded = CsvReader(in_csv, num_threads=4)

    assert_true(
        single.row_count == threaded.row_count,
        String("row_count: single={0} threaded={1}").format(
            single.row_count, threaded.row_count
        ),
    )
    assert_true(
        single.col_count == threaded.col_count,
        String("col_count: single={0} threaded={1}").format(
            single.col_count, threaded.col_count
        ),
    )
    assert_true(
        len(single) == len(threaded),
        String("length: single={0} threaded={1}").format(
            len(single), len(threaded)
        ),
    )

    var check_count = min(len(single), len(threaded))
    for i in range(check_count):
        assert_true(
            single[i] == threaded[i],
            String("element[{0}]: single='{1}' threaded='{2}'").format(
                i, single[i], threaded[i]
            ),
        )

    # Verify headers match
    assert_true(len(single.headers) == len(threaded.headers))
    for i in range(len(single.headers)):
        assert_true(
            single.headers[i] == threaded.headers[i],
            String("header[{0}]: single='{1}' threaded='{2}'").format(
                i, single.headers[i], threaded.headers[i]
            ),
        )
    print("✅ test_threaded_vs_single_threaded passed")


fn test_crlf_threaded_vs_single() raises:
    """Test CRLF consistency between single-threaded and threaded parsers."""
    # Build CSV with CRLF line endings, large enough for threading
    var csv_content = String("header1,header2,header3\r\n")
    for i in range(50):
        csv_content += String("val{0},val{0},val{0}\r\n").format(i)

    var in_csv: Path = cwd().joinpath("test_data_crlf_threaded.csv")
    in_csv.write_text(csv_content)

    var single = CsvReader(in_csv, num_threads=1)
    var threaded = CsvReader(in_csv, num_threads=4)

    assert_true(
        single.row_count == threaded.row_count,
        String("crlf row_count: single={0} threaded={1}").format(
            single.row_count, threaded.row_count
        ),
    )
    assert_true(
        len(single) == len(threaded),
        String("crlf length: single={0} threaded={1}").format(
            len(single), len(threaded)
        ),
    )

    var check_count = min(len(single), len(threaded))
    for i in range(check_count):
        assert_true(
            single[i] == threaded[i],
            String("crlf element[{0}]: single='{1}' threaded='{2}'").format(
                i, single[i], threaded[i]
            ),
        )
    print("✅ test_crlf_threaded_vs_single passed")


fn main():
    try:
        test_basic_lf()
        test_crlf_single_threaded()
        test_threaded_vs_single_threaded()
        test_crlf_threaded_vs_single()
        print("\n✅ All CI tests PASSED")
    except:
        print("❌ CI test FAILED")