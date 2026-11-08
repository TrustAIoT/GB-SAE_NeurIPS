"""Reusable YAML field groups; each experiment owns its schema and orchestration."""
from gbsae.config_cli import ConfigSchema, parse_config_arguments
import torch


def add_training_fields(parser):
    group = parser.add_group('classification training')
    group.add_field('epochs', path='training.epochs', type=int)
    group.add_field('lr', path='training.lr', type=float)
    group.add_field('weight_decay', path='training.weight_decay', type=float)
    group.add_field('batch_size', path='loader.batch_size', type=int)
    group.add_field('num_workers', path='loader.num_workers', type=int)
    group.add_field('max_batches', path='evaluation.max_batches', type=int, nullable=True)
    group.add_field('num_classes', path='training.num_classes', type=int, nullable=True)
    group.add_field('save_every', path='training.save_every', type=int)
    group.add_field('device', path='runtime.device', inherit=('base', 'runtime.device'), type=str)
    return parser


def add_activation_fields(parser):
    group = parser.add_group('frozen activation inputs')
    
    group.add_field('ckpt', path='inputs.ckpt', type=str)
    group.add_field('sources', path='selection.sources', nargs='+', nullable=True, type=str)
    group.add_field('std_mode', path='evaluation.std_mode', choices=['ckpt', 'recompute'], type=str)
    group.add_field('imagenet_root', path='data.imagenet_root', nullable=True, inherit=('base', 'data.imagenet_root'), type=str)
    group.add_field('activation_root', path='data.activation_root', nullable=True, inherit=('base', 'data.activation_root'), type=str)
    group.add_field('split', path='data.split', nullable=True, inherit=('base', 'data.split'), type=str)
    return parser
