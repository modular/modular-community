"""Test that mojo-asciichart package is installed and functional."""

from asciichart import plot

fn main() raises:
    # Test basic plotting with simple data
    var data = List[Float64]()
    data.append(1.0)
    data.append(2.0)
    data.append(4.0)
    data.append(3.0)
    data.append(5.0)
    
    # Generate chart - should not crash
    var chart = plot(data)
    
    # Basic validation: chart should contain box-drawing chars
    if len(chart) < 10:
        raise Error("Chart output too short")
    
    # Should contain at least one box-drawing character
    var has_box_char = False
    for i in range(len(chart)):
        var ch = chart[i]
        if ch == "─" or ch == "│" or ch == "┤" or ch == "├" or ch == "╭" or ch == "╯":
            has_box_char = True
            break
    
    if not has_box_char:
        raise Error("Chart missing box-drawing characters")
    
    print("✓ All tests passed")
