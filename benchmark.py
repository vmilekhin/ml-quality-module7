import cProfile
import pstats
import statistics
import timeit
import tracemalloc


def slow(values: list[float], horizon: int) -> list[float]:
    return [sum(values) / len(values) for _ in range(horizon)]


def fast(values: list[float], horizon: int) -> list[float]:
    mean = sum(values) / len(values)
    return [mean for _ in range(horizon)]


def main() -> None:
    values = [float(i) for i in range(10000)]
    assert slow(values, 30) == fast(values, 30)
    for fn in (slow, fast):
        times = timeit.repeat(lambda f=fn: f(values, 30), repeat=5, number=100)
        tracemalloc.start()
        fn(values, 30)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(fn.__name__, statistics.median(times) / 100, "s/call", peak, "bytes")
    with cProfile.Profile() as profile:
        for _ in range(100):
            slow(values, 30)
            fast(values, 30)
    pstats.Stats(profile).sort_stats("cumulative").print_stats(12)


if __name__ == "__main__":
    main()
