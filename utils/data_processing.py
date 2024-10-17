import numpy as np


def compute_mean_and_confidence_interval(data, groupby_columns, aggregation_columns, add_upper_lower=False,
                                         dropna=False):
    """
    Takes a pandas dataframe and groups the values by the groupby_columns to aggregate the aggregation_columns.
    Each aggregation column is aggregated into mean, std, count, ci. For std and ci, additional upper and lower values
    are included (e.g. {column}_ci_upper = {column}_mean + {column}_ci).
    Note that all columns of data that are in neither of the two lists (groupby_columns, aggregation_columns) will not
    be included in the results.
    :param data: Pandas dataframe of the data to aggregate.
    :param groupby_columns: List of columns to group by.
    :param aggregation_columns: List of columns to aggregate.
    :param add_upper_lower: Whether to add mean ± deviation for ci and std.
    :param dropna: dropna argument to pandas groupby.
    :return: The aggregated pandas dataframe.
    """
    aggregations = {col: ['mean', 'std', 'count'] for col in aggregation_columns}
    data = data.groupby(groupby_columns, dropna=dropna).agg(aggregations)
    data.columns = ['_'.join(col) for col in data.columns]
    data = data.reset_index()

    for col in aggregation_columns:
        data[f'{col}_ci'] = 1.96 * data[f'{col}_std'] / np.sqrt(data[f'{col}_count'])
        del data[f'{col}_count']
    if add_upper_lower:
        data = add_upper_lower_from_mean(data, aggregation_columns, ['ci', 'std'])
    return data


def add_upper_lower_from_mean(data, value_keys, deviation_keys):
    """
    Add mean ± deviation to the dataframe for all values in value_keys and deviations in deviation_keys.
    The given dataframe needs to contain the following columns: {value_key}_mean for all value keys and
    {value_key}_{deviation_key} for all combinations of value and deviation keys.
    :param data: The dataframe containing the data.
    :param value_keys: The values for which to compute mean ± deviation.
    :param deviation_keys: The deviations for which to compute mean ± deviation.
    :return: The dataframe with two additional columns {value_key}_{deviation_key}_upper/lower per combinations of value
    and deviation keys.
    """
    for value_key in value_keys:
        for deviation_key in deviation_keys:
            mean = data[f'{value_key}_mean']
            deviation = data[f'{value_key}_{deviation_key}']
            data[f'{value_key}_{deviation_key}_upper'] = mean + deviation
            data[f'{value_key}_{deviation_key}_lower'] = mean - deviation
    return data
