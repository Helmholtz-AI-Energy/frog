import torch.utils.data
from torchvision import datasets, transforms


def get_data_augmentation(image_width, horizontal_flip=False, crop_padding=4):
    augmentations = [transforms.RandomCrop(image_width, padding=crop_padding)]
    if horizontal_flip:
        augmentations.append(transforms.RandomHorizontalFlip())
    return augmentations


def get_mnist(root, dataset_config, dataloader_kwargs, download=True, data_augmentation=True, with_val_set=True):
    augmentations = get_data_augmentation(28) if data_augmentation else []
    transform = transforms.Compose([transforms.ToTensor(), *augmentations,
                                    transforms.Normalize((0.1307,), (0.3081,))])
    original_train_set = datasets.MNIST(root, train=True, download=download, transform=transform)
    if with_val_set:
        train_set, val_set = torch.utils.data.random_split(original_train_set, [50000, 10000],
                                                           generator=torch.Generator().manual_seed(0))
    else:
        train_set, val_set = original_train_set, None
    test_set = datasets.MNIST(root, train=False, download=download, transform=transform)

    return data_loaders(train_set, val_set, test_set, dataset_config.getint('batch_size'),
                        dataset_config.getint('test_batch_size'), dataloader_kwargs)


def get_cifar10(root, dataset_config, dataloader_kwargs, download=True, data_augmentation=True, with_val_set=True):
    augmentations = get_data_augmentation(32, True) if data_augmentation else []
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(), *augmentations, normalize])

    original_train_set = datasets.CIFAR10(root, train=True, download=download, transform=transform)
    if with_val_set:
        train_set, val_set = torch.utils.data.random_split(original_train_set, [40000, 10000],
                                                           generator=torch.Generator().manual_seed(0))
    else:
        train_set, val_set = original_train_set, None
    test_set = datasets.CIFAR10(root, train=False, download=download, transform=transform)

    return data_loaders(train_set, val_set, test_set, dataset_config.getint('batch_size'),
                        dataset_config.getint('test_batch_size'), dataloader_kwargs)


def get_svhn(root, dataset_config, dataloader_kwargs, download=True, data_augmentation=True, with_val_set=True):
    normalization_mean = (0.4376821, 0.4437697, 0.47280442)
    normalization_std = (0.19803012, 0.20101562, 0.19703614)
    normalize = transforms.Normalize(mean=[x for x in normalization_mean], std=[x for x in normalization_std])
    augmentations = get_data_augmentation(32) if data_augmentation else []
    transform = transforms.Compose([transforms.ToTensor(), *augmentations, normalize])

    original_train_set = datasets.SVHN(root, split='train', download=download, transform=transform)
    if with_val_set:
        train_set, val_set = torch.utils.data.random_split(original_train_set, [47225, 26032],
                                                           generator=torch.Generator().manual_seed(0))
    else:
        train_set, val_set = original_train_set, None
    test_set = datasets.SVHN(root, split='test', download=download, transform=transform)

    return data_loaders(train_set, val_set, test_set, dataset_config.getint('batch_size'),
                        dataset_config.getint('test_batch_size'), dataloader_kwargs)


def data_loaders(train_set, val_set, test_set, batch_size, test_batch_size, dataloader_kwargs):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    train_test_loader = torch.utils.data.DataLoader(train_set, batch_size=test_batch_size, **dataloader_kwargs)
    val_loader = None if val_set is None else torch.utils.data.DataLoader(
        val_set, batch_size=test_batch_size, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, **dataloader_kwargs)
    return train_loader, train_test_loader, val_loader, test_loader


def get_dataloaders(config, dataloader_kwargs, download=True, with_val_set=True):
    root = config.get('dataset_root')
    dataset = config.dataset
    dataset_config = config.get_config('dataset')
    data_augmentation = config.get('data_augmentation', datatype=bool)

    if dataset == 'mnist':
        return get_mnist(root, dataset_config, dataloader_kwargs, download, data_augmentation, with_val_set)
    if dataset == 'cifar10':
        return get_cifar10(root, dataset_config, dataloader_kwargs, download, data_augmentation, with_val_set)
    if dataset == 'svhn':
        return get_svhn(root, dataset_config, dataloader_kwargs, download, data_augmentation, with_val_set)
    else:
        raise ValueError(f'Dataset {dataset} is currently not supported.')
