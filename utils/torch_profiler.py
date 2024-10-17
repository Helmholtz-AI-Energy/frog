import datetime

import torch


class TorchProfiler:
    def __init__(self, disable=False, print_on_exit=True, **profiler_kwargs):
        self.disable = disable
        self.print_on_exit = print_on_exit
        self.profiler_kwargs = profiler_kwargs
        self.profiler = None

    def print(self, sort_by="cpu_time_total", row_limit=20, **table_kwargs) -> None:
        if self.profiler:
            start_time = datetime.datetime.now()
            print(f'[{start_time}] Preparing print. This can take a few minutes, please wait.')
            print(self.profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit, **table_kwargs))
            end_time = datetime.datetime.now()
            print(f'[{end_time}] Print done after {self.format_duration(end_time - start_time)}')

    def __enter__(self):
        if not self.disable:
            self.profiler = torch.profiler.profile(**self.profiler_kwargs)
            self.profiler.start()
        return self

    def __exit__(self, *args):
        if self.profiler:
            start_time = datetime.datetime.now()
            print(f'[{start_time}] Stopping profiler. This can take a few minutes, please wait.')
            self.profiler.stop()
            end_time = datetime.datetime.now()
            print(f'[{end_time}] Profiler stopped after {self.format_duration(end_time - start_time)}')
        if self.print_on_exit:
            self.print()

    @staticmethod
    def format_duration(duration):
        minutes = duration // datetime.timedelta(minutes=1)
        remaining_seconds = (duration - datetime.timedelta(minutes=minutes)).total_seconds()
        return f'{minutes} min {remaining_seconds} sec'
