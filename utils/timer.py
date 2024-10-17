import time


class Timer:
    DEFAULT_OUTPUT_FORMAT = ("Elapsed time {name}: last {last_elapsed_time:.2g}s, mean {mean_elapsed_time:.2g}s, "
                             "total {total_elapsed_time:.2g}s ({count})")

    def __init__(self, name="", print_on_exit=False, output_format=None):
        self.output_format = self.DEFAULT_OUTPUT_FORMAT if output_format is None else output_format
        self.print_on_exit = print_on_exit
        self.name = name

        # the current time frame
        self.start_time = None
        self.end_time = None
        self.last_elapsed_time = None

        # aggregation over all measured time frames
        self.total_elapsed_time = 0
        self.count = 0

    @property
    def mean_elapsed_time(self):
        return self.total_elapsed_time / self.count if self.count != 0 else 0

    def reset(self):
        # reset the timer (specifically the aggregated time and count)
        self.start_time = None
        self.end_time = None
        self.last_elapsed_time = None
        self.total_elapsed_time = 0
        self.count = 0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        self.end_time = time.perf_counter()
        self.last_elapsed_time = self.end_time - self.start_time

        self.total_elapsed_time += self.last_elapsed_time
        self.count += 1

    def __str__(self) -> str:
        return self.output_format.format(
            name=self.name, last_elapsed_time=self.last_elapsed_time, mean_elapsed_time=self.mean_elapsed_time,
            total_elapsed_time=self.total_elapsed_time, count=self.count)

    def print(self) -> None:
        print(self)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        if self.print_on_exit:
            self.print()
