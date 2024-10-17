import torch


def get_subclasses(cls, recursive=True, include_self=False):
    subclasses = set(cls.__subclasses__())  # direct subclasses
    if recursive:
        subclasses = subclasses.union([recursive_subclass for direct_subclass in subclasses
                                       for recursive_subclass in get_subclasses(direct_subclass, recursive=True)])
    if include_self:
        subclasses = subclasses.add(cls)
    return subclasses


def apply_to_list(item, function):
    if isinstance(item, list):
        return [apply_to_list(x, function) for x in item]
    else:
        return function(item)


def get_available_devices(cuda=True, mps=True):
    use_cuda = cuda and torch.cuda.is_available()
    use_mps = mps and torch.backends.mps.is_available()

    return [device for (device, available) in zip(["cpu", "cuda", "mps"], [True, use_cuda, use_mps]) if available]


def get_best_device(cuda=True, mps=False):
    # note that mps does not yet entirely support forward mode ad
    device_order = ["cuda", "mps", "cpu"]
    available_devices = set(get_available_devices(cuda, mps))
    for device in device_order:
        if device in available_devices:
            return device
    raise ValueError('No device available.')
