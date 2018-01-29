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

"""Policy networks for agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import gym
import numpy as np
import tensorflow as tf

import agents

tfd = tf.contrib.distributions


# TensorFlow's default implementation of the KL divergence between two
# tf.contrib.distributions.MultivariateNormalDiag instances sometimes results
# in NaN values in the gradients (not in the forward pass). Until the default
# implementation is fixed, we use our own KL implementation.
class CustomKLDiagNormal(tfd.MultivariateNormalDiag):
  """Multivariate Normal with diagonal covariance and our custom KL code."""
  pass


@tfd.RegisterKL(CustomKLDiagNormal, CustomKLDiagNormal)
def _custom_diag_normal_kl(lhs, rhs, name=None):  # pylint: disable=unused-argument
  """Empirical KL divergence of two normals with diagonal covariance.

  Args:
    lhs: Diagonal Normal distribution.
    rhs: Diagonal Normal distribution.
    name: Name scope for the op.

  Returns:
    KL divergence from lhs to rhs.
  """
  with tf.name_scope(name or 'kl_divergence'):
    mean0 = lhs.mean()
    mean1 = rhs.mean()
    logstd0 = tf.log(lhs.stddev())
    logstd1 = tf.log(rhs.stddev())
    logstd0_2, logstd1_2 = 2 * logstd0, 2 * logstd1
    return 0.5 * (
        tf.reduce_sum(tf.exp(logstd0_2 - logstd1_2), -1) +
        tf.reduce_sum((mean1 - mean0) ** 2 / tf.exp(logstd1_2), -1) +
        tf.reduce_sum(logstd1_2, -1) - tf.reduce_sum(logstd0_2, -1) -
        mean0.shape[-1].value)


def feed_forward_gaussian(
    config, action_space, observations, unused_length, state=None):
  """Independent feed forward networks for policy and value.

  The policy network outputs the mean action and the standard deviation is
  learned as independent parameter vector.

  Args:
    config: Configuration object.
    action_space: Action space of the environment.
    observations: Sequences of observations.
    unused_length: Batch of sequence lengths.
    state: Unused batch of initial states.

  Raises:
    ValueError: Unexpected action space.

  Returns:
    Attribute dictionary containing the policy, value, and unused state.
  """
  if not isinstance(action_space, gym.spaces.Box):
    raise ValueError('Network expects continuous actions.')
  if not len(action_space.shape) == 1:
    raise ValueError('Network only supports 1D action vectors.')
  action_size = action_space.shape[0]
  init_output_weights = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_output_factor)
  before_softplus_std_initializer = tf.constant_initializer(
      np.log(np.exp(config.init_std) - 1))
  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])
  with tf.variable_scope('policy'):
    x = flat_observations
    for size in config.policy_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    mean = tf.contrib.layers.fully_connected(
        x, action_size, tf.tanh,
        weights_initializer=init_output_weights)
    std = tf.nn.softplus(tf.get_variable(
        'before_softplus_std', mean.shape[2:], tf.float32,
        before_softplus_std_initializer))
    std = tf.tile(
        std[None, None],
        [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
  with tf.variable_scope('value'):
    x = flat_observations
    for size in config.value_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  mean = tf.check_numerics(mean, 'mean')
  std = tf.check_numerics(std, 'std')
  value = tf.check_numerics(value, 'value')
  policy = CustomKLDiagNormal(mean, std)
  return agents.tools.AttrDict(policy=policy, value=value, state=state)


def feed_forward_categorical(
    config, action_space, observations, unused_length, state=None):
  """Independent feed forward networks for policy and value.

  The policy network outputs the mean action and the log standard deviation
  is learned as independent parameter vector.

  Args:
    config: Configuration object.
    action_space: Action space of the environment.
    observations: Sequences of observations.
    unused_length: Batch of sequence lengths.
    state: Unused batch of initial recurrent states.

  Raises:
    ValueError: Unexpected action space.

  Returns:
    Attribute dictionary containing the policy, value, and unused state.
  """
  init_output_weights = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_output_factor)
  if not isinstance(action_space, gym.spaces.Discrete):
    raise ValueError('Network expects discrete actions.')
  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])
  with tf.variable_scope('policy'):
    x = flat_observations
    for size in config.policy_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(
        x, action_space.n, None, weights_initializer=init_output_weights)
  with tf.variable_scope('value'):
    x = flat_observations
    for size in config.value_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  policy = tfd.Categorical(logits)
  return agents.tools.AttrDict(policy=policy, value=value, state=state)


def recurrent_gaussian(
    config, action_space, observations, length, state=None):
  """Independent recurrent policy and feed forward value networks.

  The policy network outputs the mean action and the standard deviation is
  learned as independent parameter vector. The last policy layer is recurrent
  and uses a GRU cell.

  Args:
    config: Configuration object.
    action_space: Action space of the environment.
    observations: Sequences of observations.
    length: Batch of sequence lengths.
    state: Batch of initial recurrent states.

  Raises:
    ValueError: Unexpected action space.

  Returns:
    Attribute dictionary containing the policy, value, and state.
  """
  if not isinstance(action_space, gym.spaces.Box):
    raise ValueError('Network expects continuous actions.')
  if not len(action_space.shape) == 1:
    raise ValueError('Network only supports 1D action vectors.')
  action_size = action_space.shape[0]
  init_output_weights = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_output_factor)
  before_softplus_std_initializer = tf.constant_initializer(
      np.log(np.exp(config.init_std) - 1))
  cell = tf.contrib.rnn.GRUBlockCell(config.policy_layers[-1])
  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])
  with tf.variable_scope('policy'):
    x = flat_observations
    for size in config.policy_layers[:-1]:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    x, state = tf.nn.dynamic_rnn(cell, x, length, state, tf.float32)
    mean = tf.contrib.layers.fully_connected(
        x, action_size, tf.tanh,
        weights_initializer=init_output_weights)
    std = tf.nn.softplus(tf.get_variable(
        'before_softplus_std', mean.shape[2:], tf.float32,
        before_softplus_std_initializer))
    std = tf.tile(
        std[None, None],
        [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
  with tf.variable_scope('value'):
    x = flat_observations
    for size in config.value_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  mean = tf.check_numerics(mean, 'mean')
  std = tf.check_numerics(std, 'std')
  value = tf.check_numerics(value, 'value')
  policy = CustomKLDiagNormal(mean, std)
  return agents.tools.AttrDict(policy=policy, value=value, state=state)
