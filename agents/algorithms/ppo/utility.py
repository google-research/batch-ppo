# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

import tensorflow as tf
from tensorflow.python.client import device_lib


def reinit_nested_vars(variables, indices=None):
  """Reset all variables in a nested tuple to zeros.

  Args:
    variables: Nested tuple or list of variables.
    indices: Batch indices to reset, defaults to all.

  Returns:
    Operation.
  """
  if isinstance(variables, (tuple, list)):
    return tf.group(*[
        reinit_nested_vars(variable, indices) for variable in variables])
  if indices is None:
    return variables.assign(tf.zeros_like(variables))
  else:
    zeros = tf.zeros([tf.shape(indices)[0]] + variables.shape[1:].as_list())
    return tf.scatter_update(variables, indices, zeros)


def assign_nested_vars(variables, tensors, indices=None):
  """Assign tensors to matching nested tuple of variables.

  Args:
    variables: Nested tuple or list of variables to update.
    tensors: Nested tuple or list of tensors to assign.
    indices: Batch indices to assign to; default to all.

  Returns:
    Operation.
  """
  if isinstance(variables, (tuple, list)):
    return tf.group(*[
        assign_nested_vars(variable, tensor)
        for variable, tensor in zip(variables, tensors)])
  if indices is None:
    return variables.assign(tensors)
  else:
    return tf.scatter_update(variables, indices, tensors)


def discounted_return(reward, length, discount):
  """Discounted Monte-Carlo returns."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  return_ = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur + discount * agg,
      tf.transpose(tf.reverse(mask * reward, [1]), [1, 0]),
      tf.zeros_like(reward[:, -1]), 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(return_), 'return')


def fixed_step_return(reward, value, length, discount, window):
  """N-step discounted return."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  return_ = tf.zeros_like(reward)
  for _ in range(window):
    return_ += reward
    reward = discount * tf.concat(
        [reward[:, 1:], tf.zeros_like(reward[:, -1:])], 1)
  return_ += discount ** window * tf.concat(
      [value[:, window:], tf.zeros_like(value[:, -window:]), 1])
  return tf.check_numerics(tf.stop_gradient(mask * return_), 'return')


def lambda_return(reward, value, length, discount, lambda_):
  """TD-lambda returns."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  sequence = mask * reward + discount * value * (1 - lambda_)
  discount = mask * discount * lambda_
  sequence = tf.stack([sequence, discount], 2)
  return_ = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur[0] + cur[1] * agg,
      tf.transpose(tf.reverse(sequence, [1]), [1, 2, 0]),
      tf.zeros_like(value[:, -1]), 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(return_), 'return')


def lambda_advantage(reward, value, length, discount):
  """Generalized Advantage Estimation."""
  timestep = tf.range(reward.shape[1].value)
  mask = tf.cast(timestep[None, :] < length[:, None], tf.float32)
  next_value = tf.concat([value[:, 1:], tf.zeros_like(value[:, -1:])], 1)
  delta = reward + discount * next_value - value
  advantage = tf.reverse(tf.transpose(tf.scan(
      lambda agg, cur: cur + discount * agg,
      tf.transpose(tf.reverse(mask * delta, [1]), [1, 0]),
      tf.zeros_like(delta[:, -1]), 1, False), [1, 0]), [1])
  return tf.check_numerics(tf.stop_gradient(advantage), 'advantage')


def available_gpus():
  """List of GPU device names detected by TensorFlow."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def gradient_summaries(grad_vars, groups=None, scope='gradients'):
  """Create histogram summaries of the gradient.

  Summaries can be grouped via regexes matching variables names.

  Args:
    grad_vars: List of (gradient, variable) tuples as returned by optimizers.
    groups: Mapping of name to regex for grouping summaries.
    scope: Name scope for this operation.

  Returns:
    Summary tensor.
  """
  groups = groups or {r'all': r'.*'}
  grouped = collections.defaultdict(list)
  for grad, var in grad_vars:
    if grad is None:
      continue
    for name, pattern in groups.items():
      if re.match(pattern, var.name):
        name = re.sub(pattern, name, var.name)
        grouped[name].append(grad)
  for name in groups:
    if name not in grouped:
      tf.logging.warn("No variables matching '{}' group.".format(name))
  summaries = []
  for name, grads in grouped.items():
    grads = [tf.reshape(grad, [-1]) for grad in grads]
    grads = tf.concat(grads, 0)
    summaries.append(tf.summary.histogram(scope + '/' + name, grads))
  return tf.summary.merge(summaries)


def variable_summaries(vars_, groups=None, scope='weights'):
  """Create histogram summaries for the provided variables.

  Summaries can be grouped via regexes matching variables names.

  Args:
    vars_: List of variables to summarize.
    groups: Mapping of name to regex for grouping summaries.
    scope: Name scope for this operation.

  Returns:
    Summary tensor.
  """
  groups = groups or {r'all': r'.*'}
  grouped = collections.defaultdict(list)
  for var in vars_:
    for name, pattern in groups.items():
      if re.match(pattern, var.name):
        name = re.sub(pattern, name, var.name)
        grouped[name].append(var)
  for name in groups:
    if name not in grouped:
      tf.logging.warn("No variables matching '{}' group.".format(name))
  summaries = []
  # pylint: disable=redefined-argument-from-local
  for name, vars_ in grouped.items():
    vars_ = [tf.reshape(var, [-1]) for var in vars_]
    vars_ = tf.concat(vars_, 0)
    summaries.append(tf.summary.histogram(scope + '/' + name, vars_))
  return tf.summary.merge(summaries)


def set_dimension(tensor, axis, value):
  """Set the length of a tensor along the specified dimension.

  Args:
    tensor: Tensor to define shape of.
    axis: Dimension to set the static shape for.
    value: Integer holding the length.

  Raises:
    ValueError: When the tensor already has a different length specified.
  """
  shape = tensor.shape.as_list()
  if shape[axis] not in (value, None):
    message = 'Cannot set dimension {} of tensor {} to {}; is already {}.'
    raise ValueError(message.format(axis, tensor.name, value, shape[axis]))
  shape[axis] = value
  tensor.set_shape(shape)
