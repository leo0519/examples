# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
import numpy as np
import paddle
from paddle.fluid import core
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import OutScaleForTrainingPass
from paddle.fluid.contrib.slim.quantization import AddQuantDequantPass
from paddle.fluid.contrib.slim.quantization import OutScaleForInferencePass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass

try:
    from paddle.fluid.contrib.slim.quantization import utils
    TRANSFORM_PASS_OP_TYPES = utils._weight_supported_quantizable_op_type
    QUANT_DEQUANT_PASS_OP_TYPES = utils._act_supported_quantizable_op_type
except:
    TRANSFORM_PASS_OP_TYPES = QuantizationTransformPass._supported_quantizable_op_type
    QUANT_DEQUANT_PASS_OP_TYPES = AddQuantDequantPass._supported_quantizable_op_type

TENSORRT_OP_TYPES = [
    'mul', 'conv2d', 'pool2d', 'depthwise_conv2d', 'elementwise_add',
    'leaky_relu'
]

_out_scale_op_list = [
    'conv2d',
    'depthwise_conv2d',
    'mul',
    'matmul',
    'matmul_v2',
    'relu',
    'leaky_relu',
    'relu6',
    'sigmoid',
    'tanh',
    'prelu',
    'swish',
    'softmax',
    'batch_norm',
    'layer_norm',
    'elementwise_add',
    'pool2d',
    'reshape2',
    'transpose2',
    'concat',
    'elementwise_mul',
    'scale',
    'slice',
    'hard_swish',
    'hard_sigmoid',
    'conv2d_transpose',
    'gru',
    'bilinear_interp',
    'nearest_interp',
    'trilinear_interp',
    'flatten',
    'flatten2',
    'transpose',
    'pad2d',
    'reshape',
    'layer_norm',
    'flatten_contiguous_range',
]

_quant_config_default = {
    'weight_quantize_type': 'channel_wise_abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'not_quant_pattern': ['skip_quant'],
    'quantize_op_types': TENSORRT_OP_TYPES,
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9,
    'for_tensorrt': False,
    'is_full_quantize': False
}


def _get_var_value(name, scope):
    assert scope is not None, \
        'The scope cannot be set None.'
    var_tensor = scope.var(name).get_tensor()
    return np.array(var_tensor)


def _init_var_node(var_node, value, scope, place):
    assert isinstance(value,
                      np.ndarray), 'The type of value should be numpy array.'
    assert scope is not None, \
        'The scope cannot be set None.'
    assert place is not None, \
        'The place cannot be set None.'
    tensor = scope.var(var_node.name()).get_tensor()
    tensor.set(value, place)


def quant_aware(program, place, config=None, scope=None, for_test=False):

    if config is None:
        config = _quant_config_default
    if scope is None:
        scope = paddle.static.global_scope()

    logging.info('quant_aware config %s', config)

    transform_pass_ops = []
    quant_dequant_ops = []
    for op_type in config['quantize_op_types']:
        if op_type in TRANSFORM_PASS_OP_TYPES:
            transform_pass_ops.append(op_type)
        elif op_type in QUANT_DEQUANT_PASS_OP_TYPES:
            quant_dequant_ops.append(op_type)

    graph = IrGraph(core.Graph(program.desc), for_test=for_test)
    if len(transform_pass_ops) > 0:
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            weight_bits=config['weight_bits'],
            activation_bits=config['activation_bits'],
            activation_quantize_type=config['activation_quantize_type'],
            weight_quantize_type=config['weight_quantize_type'],
            window_size=config['window_size'],
            moving_rate=config['moving_rate'],
            quantizable_op_type=transform_pass_ops,
            skip_pattern=config['not_quant_pattern'])
        transform_pass.apply(graph)

    if len(quant_dequant_ops) > 0:
        quant_dequant_pass = AddQuantDequantPass(
            scope=scope,
            place=place,
            moving_rate=config['moving_rate'],
            quant_bits=config['activation_bits'],
            skip_pattern=config['not_quant_pattern'],
            quantizable_op_type=quant_dequant_ops)
        quant_dequant_pass.apply(graph)

    scale_training_pass = OutScaleForTrainingPass(
        scope=scope,
        place=place,
        moving_rate=config['moving_rate'])
    scale_training_pass.apply(graph)

    quant_program = graph.to_program()

    return quant_program


def quant_convert(program, place, config=None, scope=None):

    if config is None:
        config = _quant_config_default
    if scope is None:
        scope = paddle.static.global_scope()

    logging.info('quant_convert config %s', config)

    graph = IrGraph(core.Graph(program.desc), for_test=True)
    out_scale_infer_pass = OutScaleForInferencePass(scope=scope)
    out_scale_infer_pass.apply(graph)
    freeze_pass = QuantizationFreezePass(
        scope=scope,
        place=place,
        weight_bits=config['weight_bits'],
        activation_bits=config['activation_bits'],
        weight_quantize_type=config['weight_quantize_type'])
    freeze_pass.apply(graph)
    freezed_program = graph.to_program()

    return freezed_program

def convert_scale_fp16(model_path):
    opt_file_name = model_path + '.pdopt'
    assert os.path.exists(opt_file_name), \
        'Optimizer file [{}] not exits'.format(opt_file_name)

    with open(opt_file_name, 'rb') as f:
        load_dict = pickle.load(f, encoding='latin1')

    save_dict = {}
    for name, value in load_dict.items():
        if 'scale' in name:
            save_dict[name.replace('.cast_fp16', '')] = value.astype(np.float32)
        else:
            save_dict[name] = value

    with open(model_path + '.pdopt', 'wb') as f:
        pickle.dump(save_dict, f, protocol=4)
