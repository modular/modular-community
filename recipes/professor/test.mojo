"""Smoke test that Professor can be imported from the installed package."""

from professor import GlobalProfiler, WallClock

comptime Profiler = GlobalProfiler[WallClock]


def main() raises:
    Profiler.start()

    var list = List[Int]()
    with Profiler.zone["append"]():
        for i in range(100_000):
            list.append(i**2)

    var sum = 0
    with Profiler.zone["sum"]():
        for element in list:
            sum += element

    Profiler.end()
    print(Profiler.report())
