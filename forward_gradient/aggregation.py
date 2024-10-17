import math

import numpy as np
import torch


def forward_gradients(tangents, directional_derivatives):
    # tangents: k x n -> k x n x 1, directional_derivatives: k x batch-size -> k x 1 x batch-size
    # output via elementwise multiplication: k x n x batch-size
    k = tangents.shape[0]
    return tangents.view(k, -1, 1) * directional_derivatives.view(k, 1, -1)


def aggregate_mean(tangents, directional_derivatives):
    return forward_gradients(tangents, directional_derivatives).mean(dim=0)


def aggregate_sum(tangents, directional_derivatives):
    return forward_gradients(tangents, directional_derivatives).sum(dim=0)


def aggregate_mean_times_dim(tangents, directional_derivatives):
    return aggregate_mean(tangents, directional_derivatives) * tangents.shape[1]


def aggregate_max(tangents, directional_derivatives):
    # select the tangent with the largest absolute directional derivative
    # if the tangents have the same length, this selects the tangent which the most with the gradient
    max_index = directional_derivatives.abs().argmax()
    return tangents[max_index] * directional_derivatives[max_index]


def aggregate_max_normalized(tangents, directional_derivatives):
    # same as max but normalize with tangent length, i.e. select tangent with max cosine sim to the gradient
    max_index = (directional_derivatives / tangents.norm(dim=1)).abs().argmax()
    return tangents[max_index] * directional_derivatives[max_index]


def aggregate_mean_of_top_k(tangents, directional_derivatives):
    # select the best tangents (using the maximum cosine similarity normalized by tangents length)
    # in contrast to max and max_normalized, multiple tangents can be selected
    max_index = (directional_derivatives / tangents.norm(dim=1)).abs().argmax()
    cosine_similarities = (directional_derivatives / tangents.norm(dim=1)).abs()
    max_distance_to_best = 0.1
    mask = cosine_similarities >= (cosine_similarities.max() - max_distance_to_best)
    return (tangents * (mask * directional_derivatives)[:, None]).mean(dim=0)


def select_tangent_subset(tangents, directional_derivatives, tangent_selection='first_n'):
    k = tangents.shape[0]
    n = np.prod(tangents.shape[1:])

    if k <= n:
        return tangents, directional_derivatives

    if tangent_selection == 'max':
        indices = torch.topk((directional_derivatives / tangents.norm(dim=1)).abs(), n).indices
    elif tangent_selection == 'min':
        indices = torch.topk((directional_derivatives / tangents.norm(dim=1)).abs(), n, largest=False).indices
    elif tangent_selection == 'first_n':
        indices = torch.arange(n)
    else:
        raise ValueError(f'Invalid {tangent_selection=}.')

    return tangents[indices], directional_derivatives[indices]


def compute_inverse_gram_matrix(basis, check_det=False, min_det=1e-10):
    # compute the gram matrix
    # k = tangents.shape[0]
    # gram_matrix = torch.matmul(tangents.view(k, -1), tangents.view(k, -1).T)
    gram_matrix = torch.matmul(basis.T, basis)

    try:  # try inverting the gram matrix
        if check_det:  # if the determinate is small the gram is likely to not be actually invertible
            gram_det = torch.linalg.det(gram_matrix)
            if gram_det.abs() < min_det:
                raise RuntimeError(f'Gram determinant {gram_det} is smaller than {min_det}. '
                                   f'Assuming Gram matrix to be irregular.')
        inverse_gram_matrix = torch.linalg.inv(gram_matrix)
    except RuntimeError as e:
        raise RuntimeError(f'Error inverting the gram matrix. Note that the tangents must be linearly independent. {e}')

    return inverse_gram_matrix


def aggregate_orthogonal_projection(tangents, directional_derivatives, check_det=False, min_det=1e-10,
                                    tangent_selection='first_n'):
    # If there are more tangents (k) than dimensions (n) the tangents cannot be linearly independent, thus the gram
    # matrix cannot be invertible. In that case, select a subset of n tangents and corresponding derivatives.
    tangents, directional_derivatives = select_tangent_subset(tangents, directional_derivatives, tangent_selection)

    # computes the orthogonal projection B(B^⊤B)⁻¹B^⊤∇f of the gradient ∇f onto the subspace spanned by the
    # directions vᵢ with B=(v₁|...|vₖ) = directions_v^⊤. B^⊤∇f corresponds to the directional derivatives.
    # assumes the directions vᵢ are a basis, i.e., linearly independent. Will throw an error otherwise.
    basis = tangents.T  # tangents are assumed to be flattened, yielding B^⊤, transpose for B

    inverse_gram_matrix = compute_inverse_gram_matrix(basis, check_det, min_det)  # compute G⁻¹ = (B^⊤B)⁻¹
    # compute B(G⁻¹B^⊤∇f) in that order (i.e. G⁻¹B^⊤∇f first) as its more computationally efficient since k ≤ n
    return torch.matmul(basis, torch.matmul(inverse_gram_matrix, directional_derivatives))


aggregation_methods = {
    'mean': aggregate_mean,
    'sum': aggregate_sum,
    'mean_times_dim': aggregate_mean_times_dim,
    'max': aggregate_max,
    'max_normalized': aggregate_max_normalized,
    'mean_of_top_k': aggregate_mean_of_top_k,
    'orthogonal_projection': aggregate_orthogonal_projection,
}


def chi_expected_value(k, approx_larger_than=500):
    # compute the expected value of the chi distribution χₖ
    if approx_larger_than and k > approx_larger_than:  # approximate for large k using lgamma
        # y = gamma(a) / gamma(b) = e ^ log(y) with log(y) = log(gamma(a))- log(gamma(b))
        return math.sqrt(2) * math.exp(math.lgamma((k + 1) / 2) - math.lgamma(k / 2))
    return math.sqrt(2) * math.gamma((k + 1) / 2) / math.gamma(k / 2)


def scaling_correction_factor(aggregation, k, n, expected_tangent_length=None):
    # TODO: it should be sufficient to compute this correction factor once and reuse it while the parameters
    # remain the same. But: where to store it and how to incorporate reuse into the training process?
    if aggregation == 'orthogonal_projection':
        expected_length = 1 if k >= n else chi_expected_value(k) / chi_expected_value(n)
        return 1 / expected_length
    elif aggregation == 'mean':
        squared_tangent_length = (expected_tangent_length or 1) ** 2  # expected |v|^2
        angle_and_addition_factor = math.sqrt(2 / (math.pi * n)) if k == 1 else math.sqrt((n + k) / (1 + k)) * 1 / n
        expected_length = squared_tangent_length * angle_and_addition_factor
        return 1 / expected_length
    elif aggregation == 'sum':
        # FG-mean = FG-sum / k -> correction_factor(sum) = 1 / k * correction_factor(mean)
        return 1 / k * scaling_correction_factor('mean', k, n, expected_tangent_length)
    else:
        # TODO: could simulate in other cases (probably need at least a few samples to be sufficiently accurate)
        # but that should really be done only once
        raise ValueError(f'{aggregation=} currently not supported.')


def forward_gradient(tangents, directional_derivatives, aggregation, scaling_correction=False,
                     expected_tangent_length_fn=None, correction_factor=None, **kwargs):
    if aggregation not in aggregation_methods:
        raise ValueError(f'Invalid {aggregation=}. Available aggregation methods: '
                         f'{", ".join(str(key) for key in aggregation_methods)}')

    # flatten secondary dimensions
    flat_tangents = tangents.view(tangents.shape[0], -1)

    # compute forward gradient
    flat_forward_gradient = aggregation_methods[aggregation](flat_tangents, directional_derivatives, **kwargs)

    # optionally: apply scaling correction
    if scaling_correction:
        if correction_factor is None:
            k, n = flat_tangents.shape
            expected_tangent_length = None if expected_tangent_length_fn is None else expected_tangent_length_fn(n)
            correction_factor = scaling_correction_factor(aggregation, k, n, expected_tangent_length)
        flat_forward_gradient *= correction_factor

    # undo the flattening and reshape to original shape
    batch_size = directional_derivatives.shape[-1] if len(directional_derivatives.shape) > 1 else None
    grad_shape = tangents.shape[1:] if batch_size is None else [*tangents.shape[1:], batch_size]
    return flat_forward_gradient.view(grad_shape)
