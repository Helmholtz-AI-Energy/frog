import argparse
import pathlib

import pandas
import torch
import tqdm.auto

from utils import compute_mean_and_confidence_interval, construct_output_path
from forward_gradient import tangent_sampling, aggregation


def compare_vectors(a, b):
    """
    Compare the two vectors a and b of size n. Vector b can optionally be batches, i.e. of shape m x n where n is the
    vector dimension and m the optional batch dimension. All values are computed for each vector in the batch.
    :param a: An 1 x n-dimensional vector.
    :param b: An m x n-dimensional vector.
    :return: An m x 6 tensor containing the following values in each row:
        a_norm: The vector norm of a (repeated for all m).
        b_norm: The vector norm of b.
        difference_norm: The vector norm of a - b.
        a_norm_div_b_norm: The vector norm of b divided by that of a.
        dot_product: The dot product of a and b.
        cosine_sim: The cosine similarity of a and b.
    """
    a_norm = torch.linalg.vector_norm(a, dim=1)  # (1,)
    b_norm = torch.linalg.vector_norm(b, dim=1)  # (m,)
    difference_norm = torch.linalg.vector_norm(a - b, dim=1)  # (m,)
    a_norm_div_b_norm = b_norm / a_norm  # (m,)

    dot_product = torch.matmul(a, b.T).view_as(b_norm)  # (m,)
    cosine_sim = dot_product / (a_norm * b_norm)  # (m,)

    return torch.stack([a_norm.expand(b_norm.shape[0]), b_norm, difference_norm,
                        a_norm_div_b_norm, dot_product, cosine_sim], dim=1)


def compute_forward_gradients(true_gradient, num_directions, aggregation_mode, aggregation_kwargs=None,
                              tangent_sampler='normal', tangent_postprocessing=None, tangent_sampling_kwargs=None,
                              num_samples=1000, seed=0, device='cpu', progressbar=None):
    aggregation_kwargs = aggregation_kwargs or {}
    tangent_sampling_kwargs = tangent_sampling_kwargs or {}

    true_gradient = true_gradient.to(device)
    n = true_gradient.shape[0]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    forward_gradients = []

    # note that we cannot compute multiple samples in a batched fashion as the forward gradient computation flattens all
    # but the first tangent dimension to make handling network weights of varying shapes easier.
    for _ in range(num_samples):
        # create tangents of shape k x n
        tangents = tangent_sampling.sample_tangents(
            (num_directions, n), sampler=tangent_sampler, postprocessing=tangent_postprocessing, generator=generator,
            device=device, **tangent_sampling_kwargs)

        # compute jvps as dot product with gradient (i.e. sim-FG approach instead of FwAD)
        jvps = torch.matmul(tangents, true_gradient)

        forward_gradient = aggregation.forward_gradient(tangents, jvps, aggregation_mode, **aggregation_kwargs)
        forward_gradients.append(forward_gradient)  # a n-dimensional vector
        if progressbar is not None:
            progressbar.update()
    return torch.stack(forward_gradients, dim=0)  # n x samples dimensional tensor


def normalize(x):
    return x / torch.norm(x)


true_grad_fns = {
    'ones': lambda n: torch.ones(n),
    'ones_normalized': lambda n: normalize(torch.ones(n)),
    'random_unnormalized': lambda n: torch.randn(n),
    'random_normalized': lambda n: normalize(torch.randn(n)),
}


def frog_dicts(k, scaling_correction=False, tangent_sampler='normal'):
    label_dict = dict(k=k, agg='orthogonal_projection', tangents=tangent_sampler, scaling_correction=scaling_correction,
                      label=f'frog_{k=}', normalize_tangents=False)
    config_dict = dict(num_directions=k, aggregation_mode='orthogonal_projection',
                       aggregation_kwargs={'scaling_correction': scaling_correction}, tangent_sampler=tangent_sampler)
    return label_dict, config_dict


def conical_fg_dicts(k, agg, scaling_correction=False, tangent_sampler='normal', normalize_tangents=False):
    label_dict = dict(k=k, agg=agg, tangents=tangent_sampler, scaling_correction=scaling_correction,
                      normalize_tangents=normalize_tangents, label=f'fg_{agg}_{k=}')
    tangent_postprocessing = [tangent_sampling.normalize] if normalize_tangents else []
    config_dict = dict(num_directions=k, aggregation_mode=agg,
                       aggregation_kwargs={'scaling_correction': scaling_correction}, tangent_sampler=tangent_sampler,
                       tangent_postprocessing=tangent_postprocessing)
    return label_dict, config_dict


def baseline_dicts(scaling_correction=False, tangent_sampler='normal', normalize_tangents=False):
    label_dict, config_dict = conical_fg_dicts(1, 'sum', scaling_correction, tangent_sampler, normalize_tangents)
    label_dict['agg'] = 'none'
    label_dict['label'] = 'fg_baseline'
    return label_dict, config_dict


def compare_forward_gradients_to_true_gradient(true_gradient_fn, forward_gradient_configs, dimensions,
                                               num_samples=1000, device='cpu', **kwargs):
    """
    Compare the forward gradients to the true gradient to evaluate the approximation quality.
    :param true_gradient_fn: The approach to compute true gradients, must be a key in true_grad_fns.
    :param forward_gradient_configs: A list of label_dict, config_dict pairs for each of the forward gradient approaches
    to test. label_dict contains the key-value pairs used to label the results in the results dataframe, while
    config_dict contains the corresponding parameters to compute_forward_gradients.
    :param dimensions: A list of dimensions to compute gradients for.
    :param num_samples: The number of forward gradient samples per combination.
    :param device: The device to perform the computation on.
    :param kwargs: Additional kwargs to compute_forward_gradients.
    :return: Pandas DataFrame of the comparison results.
    """
    results = []
    progressbar = tqdm.auto.tqdm(total=len(dimensions) * len(forward_gradient_configs) * num_samples)

    for n in dimensions:
        # compute/sample true gradient
        true_gradient = true_grad_fns[true_gradient_fn](n)
        true_gradient = true_gradient.to(device)
        # for each forward gradient approach
        for fg_label_dict, fg_config_dict in forward_gradient_configs:
            # compute <num_samples> forward gradients
            forward_gradients = compute_forward_gradients(true_gradient, **fg_config_dict, num_samples=num_samples,
                                                          progressbar=progressbar, device=device, **kwargs)
            # compare all forward gradients to the true gradient
            comparison_results = compare_vectors(true_gradient.view(1, -1), forward_gradients)

            # build pandas dataframe for results
            df_value_columns = ['true_grad_norm', 'fg_norm', 'difference_norm',
                                'fg_norm/true_grad_norm', 'dot_product', 'cosine_sim']
            results_df = pandas.DataFrame(comparison_results.cpu().numpy(), columns=df_value_columns)
            results_df['n'] = n
            for key, value in fg_label_dict.items():
                results_df[key] = value
            if 'tangent_sampling_kwargs' in kwargs:
                for key, value in kwargs['tangent_sampling_kwargs'].items():
                    results_df[f'tangent_{key}'] = value
            # collect partial results df in global list
            results.append(results_df)

    # combine all partial results dfs
    all_results = pandas.concat(results)
    all_results['true_grad_fn'] = true_gradient_fn
    return all_results


def approximation_quality_experiments(ns, ks, true_gradient_fn='ones', tangent_sampler='normal', save_to=None,
                                      **kwargs):
    forward_gradient_configs = [baseline_dicts(tangent_sampler=tangent_sampler)] + [
        frog_dicts(k, tangent_sampler=tangent_sampler) for k in ks] + [
        conical_fg_dicts(k, agg, tangent_sampler=tangent_sampler, normalize_tangents=normalize_tangents)
        for agg in ['mean', 'sum'] for normalize_tangents in [True, False] for k in ks]
    results_df = compare_forward_gradients_to_true_gradient(true_gradient_fn, forward_gradient_configs, ns, **kwargs)

    # 'true_grad_norm' is intentionally used as groupby and not aggregation column as it should be constant for each
    # group, so aggregating it just adds noise
    aggregation_columns = ['fg_norm', 'difference_norm', 'fg_norm/true_grad_norm', 'dot_product', 'cosine_sim']
    groupby_cols = [col for col in results_df.columns if col not in aggregation_columns]
    summary_df = compute_mean_and_confidence_interval(results_df, groupby_cols, aggregation_columns)
    if save_to:
        print(f'Writing results to {save_to}')
        results_df.to_csv(save_to / 'raw_data.csv')
        summary_df.to_csv(save_to / 'summary.csv')
    return results_df, summary_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tangents', type=str, default='normal', choices=['normal', 'angle'],
                        help='How the tangents are sampled (either from a normal distribution or with specific angle '
                             'to the first tangent)')
    parser.add_argument('--angles', type=int, nargs='+',
                        help='Angle of secondary tangents to first tangent when using --tangents angle')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples per gradient approximation.')
    cli_args = parser.parse_args()

    output_base_path = pathlib.Path(__file__).parent / 'results' / 'approximation_quality'

    if cli_args.tangents == 'normal':
        ns = [2**i for i in range(15)]  # for our experiments, we used num_samples=1000 up to n=2**14
        # ns = [2**i for i in range(15, 17)]  # and num_samples=100 for n >= 2**15
        ks = [2**i for i in range(11)]
        output_path = construct_output_path(output_base_path, 'approx_quality')
        output_path.mkdir()
        approximation_quality_experiments(ns, ks, save_to=output_path, num_samples=cli_args.num_samples)
    else:
        ns = [64]
        ks = [4, 16, 64]
        tangent_options = {'angle': ('specific_angle', cli_args.angles)}
        tangent_sampler, values = tangent_options[cli_args.tangents]
        for value in values:
            output_path = construct_output_path(output_base_path, f'approx_quality__{cli_args.tangents}__{value}')
            output_path.mkdir()
            approximation_quality_experiments(ns, ks, save_to=output_path, num_samples=cli_args.num_samples,
                                              tangent_sampler=f'{tangent_sampler}_{value}')
