import argparse
import configparser
import pathlib
import sys

from utils import construct_output_path, append_suffix, get_best_device

DEFAULT_CONFIG_FILE_LOCATION = pathlib.Path(__file__).parent / 'configs'
DEFAULT_CONFIG_FILES = DEFAULT_CONFIG_FILE_LOCATION / 'defaults.ini'


class Configuration:
    base_config_keys = ['dataset', 'gradient_computation', 'perturbation_mode', 'model', 'optimizer']

    def __init__(self, args):
        self.config_parser = configparser.ConfigParser()
        self.read(args.configfile)

        self.dataset = args.dataset
        self.gradient_computation = args.gradient_computation
        self.perturbation_mode = args.perturbation_mode
        self.model = args.model
        self.optimizer = args.optimizer
        self.progress_bar = args.show_progress

        # overwrite defaults/configfiles
        if args.epochs is not None:
            self.set('epochs', args.epochs)
        else:
            dataset_epochs = self.get('epochs', section=self.dataset)
            if dataset_epochs is not None:
                self.set('epochs', dataset_epochs)
        if args.seed is not None:
            self.set('seed', args.seed)
        if args.initial_lr is not None:
            self.set('initial_lr', args.initial_lr, self.optimizer)
        if args.num_directions is not None:
            self.set('num_directions', args.num_directions, self.gradient_computation)
        if args.fg_computation_mode is not None:
            self.set('fg_computation_mode', args.fg_computation_mode, self.gradient_computation)
        if args.model_hidden_size is not None:
            self.set('hidden_size', args.model_hidden_size, 'FC')
        if args.output_name is not None:
            self.set('output_name', args.output_name)
        if args.experiment_id is not None:
            self.set('experiment_id', args.experiment_id)

        self.device = get_best_device() if args.device is None else args.device
        self.set('device', self.device)

        # add remaining args to config object
        self.config_parser.add_section('BASECONFIG')
        self.set('dataset', self.dataset, 'BASECONFIG')
        self.set('gradient_computation', self.gradient_computation, 'BASECONFIG')
        self.set('perturbation_mode', self.perturbation_mode, 'BASECONFIG')
        self.set('model', self.model, 'BASECONFIG')
        self.set('optimizer', self.optimizer, 'BASECONFIG')

        # construct and set output_path
        self.output_path = None
        self.update_output_path()

        # clean up config object: delete unused sections
        used_sections = [self.get(key, section='BASECONFIG').upper()
                         for key in self.base_config_keys] + ['DEFAULT', 'BASECONFIG']
        unused_sections = [s for s in self.config_parser.sections() if s not in used_sections]
        for unused_section in unused_sections:
            self.config_parser.remove_section(unused_section)

    def update_output_path(self):
        # construct and set output path from output_base, output_name, experiment_id values
        self.output_path = construct_output_path(*[self.get(key) for key in ['output_base', 'output_name',
                                                                             'experiment_id']]).absolute()
        self.set('output_path', self.output_path)

    def get_config(self, subconfig_type):
        section_name = self.get(subconfig_type, section='BASECONFIG')
        return self.config_parser[section_name.upper()]

    def get_base_config_dict(self):
        return {key: self.get(key, section='BASECONFIG') for key in self.base_config_keys}

    def read(self, *configfiles, include_default=True):
        if include_default:
            configfiles = [DEFAULT_CONFIG_FILES, *configfiles]
        configfiles = [str(file) for file in configfiles]
        print(f'Reading from {configfiles}', file=sys.stderr)
        self.config_parser.read([str(file) for file in configfiles])

    def save(self, output_path=None):
        output_path = append_suffix(self.output_path if output_path is None else output_path, '.ini')
        print(f'Saving config to {output_path.absolute()}', file=sys.stderr)
        with open(output_path, 'w') as configfile:
            self.config_parser.write(configfile)

    def get(self, key, fallback=None, section=None, datatype=None):
        section = (section or 'DEFAULT').upper()
        return self.get_from_config(self.config_parser[section], key, fallback, datatype)

    @staticmethod
    def get_from_config(config, key, fallback=None, datatype=None):
        if datatype in ['int', int]:
            return config.getint(key, fallback)
        elif datatype in ['float', float]:
            return config.getfloat(key, fallback)
        elif datatype in ['boolean', 'bool', bool]:
            return config.getboolean(key, fallback)
        else:
            return config.get(key, fallback)

    @staticmethod
    def get_multiple_from_config(config, keys_and_types, skip_nones=True):
        keys_and_values = {}
        for key_and_type in keys_and_types:
            # extract key and datatype, if datatype is not available, set to None
            try:
                key, datatype, fallback, *_ = list(key_and_type) + [None] * 3
            except TypeError:
                key, datatype, fallback = key_and_type, None, None

            value = Configuration.get_from_config(config, key, datatype=datatype, fallback=fallback)
            if value is not None or not skip_nones:
                if key in keys_and_values:
                    print(f'Warning: overwriting value for {key=} in Configuration.get_multiple_from_config. '
                          f'Old value {keys_and_values[key]}, new value {value}.', file=sys.stderr)
                keys_and_values[key] = value
        return keys_and_values

    def set(self, key, value, section=None):
        self.config_parser[(section or 'DEFAULT').upper()][key] = str(value)


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'svhn'])
    parser.add_argument('--gradient_computation', type=str, default='bp', choices=['bp', 'fg', 'frog'])
    parser.add_argument('--perturbation_mode', type=str, default='node', choices=['weight', 'node'])
    parser.add_argument('--model', type=str, default='fc', choices=['fc', 'lenet5', 'resnet18', 'vit', 'mlpmixer'])
    parser.add_argument('--model_hidden_size', type=int, help='Overwrite hidden size of fc model.')
    parser.add_argument('--optimizer', type=str, default='plain_sgd', choices=['plain_sgd', 'sota_sgd', 'sota_adam'])
    parser.add_argument('--configfile', type=pathlib.Path, default=DEFAULT_CONFIG_FILES)
    parser.add_argument('--fg_computation_mode', type=str, choices=['fwad', 'sim'],
                        help='Whether to use true forward-mode AD to compute forward gradients ("fwad") or to simulate'
                             'them via backpropagation ("sim").')

    parser.add_argument('--device', type=str, help='Select specific torch device.')
    parser.add_argument('--show_progress', action='store_true', help='Activate progress bar.')

    parser.add_argument('--experiment_id', type=str)
    parser.add_argument('--output_name', type=str)

    parser.add_argument('--epochs', type=int,
                        help='Select specific number of epochs (overwrites config file).')
    parser.add_argument('--seed', type=int, help='Select specific seed (overwrites config file).')
    parser.add_argument('--initial_lr', type=float, help='Select specific lr (overwrites config file).')
    parser.add_argument('--num_directions', type=int,
                        help='Select specific number of random directions for forward gradients '
                             '(overwrites config file).')

    return parser
