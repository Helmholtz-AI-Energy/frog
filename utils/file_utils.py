import datetime
import pathlib
import re
import sys
import uuid


def path_friendly_string(s):
    return re.sub(r"\s+", '_', s).lower()


def construct_output_path(output_path=".", output_name="", experiment_id=None, mkdir=True):
    today = datetime.datetime.today()
    path = pathlib.Path(output_path) / str(today.year) / f"{today.year}-{today.month}"
    if experiment_id:
        path = path / experiment_id / str(today.date())
    else:
        path /= str(today.date())
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    base_filename = f"{today.strftime('%Y-%m-%d--%H-%M-%S')}-{output_name}-{str(uuid.uuid4())[:8]}"
    return path / base_filename


def append_suffix(path, suffix):
    path = pathlib.Path(path)
    if path.suffix != suffix:
        path = path.parent / (path.name + suffix)
    return path


def save_dataframe(dataframe, config, append_baseconfig=True):
    output_path = append_suffix(config.output_path, '_train_progress.csv')
    print(f"Saving results to {output_path.absolute()}", file=sys.stderr)
    dataframe["result_filename"] = output_path
    if append_baseconfig:
        for key, value in config.get_base_config_dict().items():
            dataframe[key] = value
    dataframe.to_csv(output_path, index=False)