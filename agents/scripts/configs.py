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

"""Example configurations using the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-variable

import tensorflow as tf

from agents import algorithms
from agents.scripts import networks


def default():
  """Default configuration for PPO."""
  # General
  algorithm = algorithms.PPO
  num_agents = 30
  eval_episodes = 30
  use_gpu = False
  # Environment
  normalize_ranges = True
  # Network
  network = networks.feed_forward_gaussian
  weight_summaries = dict(
      all=r'.*', policy=r'.*/policy/.*', value=r'.*/value/.*')
  policy_layers = 200, 100
  value_layers = 200, 100
  init_output_factor = 0.1
  init_std = 0.35
  # Optimization
  update_every = 30
  update_epochs = 25
  optimizer = tf.train.AdamOptimizer
  learning_rate = 1e-4
  # Losses
  discount = 0.995
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  return locals()


def pendulum():
  """Configuration for the pendulum classic control task."""
  locals().update(default())
  # Environment
  env = 'Pendulum-v0'
  max_length = 200
  steps = 1e6  # 1M
  # Optimization
  batch_size = 20
  chunk_length = 50
  return locals()


def cartpole():
  """Configuration for the cart pole classic control task."""
  locals().update(default())
  # Environment
  env = 'CartPole-v1'
  max_length = 500
  steps = 2e5  # 200k
  normalize_ranges = False  # The env reports wrong ranges.
  # Network
  network = networks.feed_forward_categorical
  return locals()


def reacher():
  """Configuration for MuJoCo's reacher task."""
  locals().update(default())
  # Environment
  env = 'Reacher-v2'
  max_length = 1000
  steps = 5e6  # 5M
  discount = 0.985
  update_every = 60
  return locals()


def cheetah():
  """Configuration for MuJoCo's half cheetah task."""
  locals().update(default())
  # Environment
  env = 'HalfCheetah-v2'
  max_length = 1000
  steps = 1e7  # 10M
  discount = 0.99
  return locals()


def walker():
  """Configuration for MuJoCo's walker task."""
  locals().update(default())
  # Environment
  env = 'Walker2d-v2'
  max_length = 1000
  steps = 1e7  # 10M
  return locals()


def hopper():
  """Configuration for MuJoCo's hopper task."""
  locals().update(default())
  # Environment
  env = 'Hopper-v2'
  max_length = 1000
  steps = 1e7  # 10M
  update_every = 60
  return locals()


def ant():
  """Configuration for MuJoCo's ant task."""
  locals().update(default())
  # Environment
  env = 'Ant-v2'
  max_length = 1000
  steps = 2e7  # 20M
  return locals()


def humanoid():
  """Configuration for MuJoCo's humanoid task."""
  locals().update(default())
  # Environment
  env = 'Humanoid-v2'
  max_length = 1000
  steps = 5e7  # 50M
  update_every = 60
  return locals()


def bullet_ant():
  """Configuration for PyBullet's ant task."""
  locals().update(default())
  # Environment
  import pybullet_envs  # noqa pylint: disable=unused-import
  env = 'AntBulletEnv-v0'
  max_length = 1000
  steps = 3e7  # 30M
  update_every = 60
  return locals()
