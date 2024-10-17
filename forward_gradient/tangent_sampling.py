import functools
import re

import numpy as np
import torch


def rademacher(size, *, generator=None, device='cpu'):
    return torch.randint(high=2, size=size, generator=generator, device=device) * 2. - 1.


def standard_basis(size, *, generator=None, device='cpu'):
    k = size[0]
    n = np.prod(size[1:])
    # create n x k, then transpose to k x n because additional columns are filled with 0
    eye = torch.eye(n, k, device=device).T
    # tangents are the first k standard basis vectors of dimension n, reshaped to the tangent shape
    return eye.view(size)


def reject_vectors(base, vectors_to_reject):
    # compute vector rejection of one or multiple vectors_to_reject from the base vector
    normalized_base = base / torch.linalg.vector_norm(base, dim=1)
    vector_projections = torch.matmul(normalized_base, vectors_to_reject.T).T * normalized_base
    return vectors_to_reject - vector_projections


def rotate_within_plane(base_tangent, secondary_tangents, angle_to_base):
    angle_to_base_rad = torch.tensor(angle_to_base).deg2rad()
    normalized_base = base_tangent / torch.linalg.vector_norm(base_tangent, dim=1)
    first_summand = angle_to_base_rad.cos() * normalized_base

    vector_rejections = reject_vectors(normalized_base, secondary_tangents)
    normalized_vector_rejections = vector_rejections / torch.linalg.vector_norm(vector_rejections, dim=1).unsqueeze(-1)
    second_summands = angle_to_base_rad.sin() * normalized_vector_rejections
    return first_summand + second_summands


def tangents_with_specific_angle(size, *, angle, generator=None, device='cpu'):
    k, n = size
    sampled_tangents = torch.randn(k, n, generator=generator, device=device)
    normalized_tangents = sampled_tangents / torch.linalg.vector_norm(sampled_tangents, dim=1, keepdim=True)
    base_tangent = normalized_tangents[0:1]
    secondary_tangents = normalized_tangents[1:]
    rotated_tangents = rotate_within_plane(base_tangent, secondary_tangents, angle) if angle else secondary_tangents
    return torch.cat([base_tangent, rotated_tangents])


def orthogonal_tangents_of_different_lengths(size, *, max_length, generator=None, device='cpu'):
    k, n = size
    # tangent lengths are uniformly distributed on [1, max_scale]
    tangent_lengths = (max_length - 1) * torch.rand(k, 1, generator=generator, device=device) + 1
    # tangents = k random standard basis vectors but with different scales
    return torch.eye(n, device=device)[torch.randperm(n, generator=generator)[:k]] * tangent_lengths


samplers = {
    'rademacher': rademacher,
    'normal': torch.randn,
    'eye': standard_basis,
    'standard_basis': standard_basis,
}


def get_sampler(key):
    if key in samplers:
        return samplers[key]
    if match := re.search(r'specific_angle_(?P<angle>.+)', key):
        return functools.partial(tangents_with_specific_angle, angle=int(match.group('angle')))
    elif match := re.search(r'varying_length_(?P<scale>.+)', key):
        return functools.partial(orthogonal_tangents_of_different_lengths, max_length=float(match.group('scale')))
    raise ValueError(f'No matching tangent sampler found for {key=}.')


def secondary_dimensions_independent(tensor):
    # Checks if the k x n - matrix obtained by flattening all but the first dimension has rank == k = first dimension.
    # If the flattened secondary dimensions n are at least as large as the first dimension k (i.e. k ≤ n), this
    # corresponds to the matrix having full rank. Note that if k > n this will always be false as the matrix can have
    # at most rank n.
    k = tensor.shape[0]
    n = np.prod(tensor.shape[1:])
    return k <= n and (torch.linalg.matrix_rank(tensor.view(k, n)) == k)


def normalize(tensor):
    secondary_dimensions = list(range(1, len(tensor.shape)))  # all but the first dimension
    return tensor / tensor.norm(dim=secondary_dimensions, keepdim=True)


def sample_tangents(size, *, sampler, postprocessing, generator=None, resample=0, device='cpu'):
    sampler_fn = get_sampler(sampler)
    tangents = sampler_fn(size, generator=generator, device=device)

    # recursive resampling until resample <= 0 or the tangents are linearly independent (have rank at least k)
    if resample > 0 and not secondary_dimensions_independent(tangents):
        return sample_tangents(size, sampler=sampler, postprocessing=postprocessing, generator=generator,
                               resample=resample - 1, device=device)

    if postprocessing:  # changed application f3(f2(f1(tangents))) of postprocessing functions f1, f2, f3,...
        tangents = functools.reduce(lambda x, fn: fn(x), postprocessing, tangents)

    return tangents
