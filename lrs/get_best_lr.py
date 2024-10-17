import pathlib

import numpy as np
import pandas

from function_optimization import gradient_label

LR_CSV_PATH = pathlib.Path(__file__).parent
BEST_LRS_CSV = {
    'math': LR_CSV_PATH / 'math.csv',
    'math_custom_tangents': LR_CSV_PATH / 'math__custom_tangents.csv',
    'fc_nns': LR_CSV_PATH / 'fc_nns.csv',
    'sota_nns': LR_CSV_PATH / 'sota_nns.csv',
}


def get_lr_from_csv(csv_path, config):
    lr_df = pandas.read_csv(csv_path)
    index_columns = list(config.keys())
    lr_df.set_index(index_columns, inplace=True)
    index = tuple(config[column] for column in index_columns)
    return lr_df.loc[index, 'lr']


def get_math_lr(csv_path, function, n, gradient_approach):
    task_config = {'function': function, 'n': n}
    approach_config = {'optimizer': gradient_label(gradient_approach, ignore_keys=['scaling_correction'])}
    try:
        return get_lr_from_csv(csv_path, {**task_config, **approach_config})
    except KeyError as e:
        gradient_config = gradient_approach.get_config()
        if gradient_config.get('direction_distribution', 'normal') != 'normal':
            fallback_config = gradient_approach.get_config()
            fallback_config['direction_distribution'] = 'normal'
            del fallback_config['type']
            return get_math_lr(csv_path, function, n, type(gradient_approach)(**fallback_config))
        raise e


def get_nn_lr(csv_path, model, dataset, gradient_approach, k=np.nan):
    task_config = {'model': model, 'dataset': dataset}
    approach_config = {'gradient_approach': gradient_approach, 'k': k}
    return get_lr_from_csv(csv_path, {**task_config, **approach_config})
