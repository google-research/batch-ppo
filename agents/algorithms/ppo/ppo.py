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

"""Proximal Policy Optimization agent.

Based on John Schulman's implementation in Python and Theano:
https://github.com/joschu/modular_rl/blob/master/modular_rl/ppo.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from agents import parts
from agents import tools
from agents.algorithms.ppo import utility


class PPO(object):
  """A vectorized implementation of the PPO algorithm by John Schulman."""

  def __init__(self, batch_env, step, is_training, should_log, config):
    """Create an instance of the PPO algorithm.

    Args:
      batch_env: In-graph batch environment.
      step: Integer tensor holding the current training step.
      is_training: Boolean tensor for whether the algorithm should train.
      should_log: Boolean tensor for whether summaries should be returned.
      config: Object containing the agent configuration as attributes.
    """
    self._batch_env = batch_env
    self._step = step
    self._is_training = is_training
    self._should_log = should_log
    self._config = config
    self._observ_filter = parts.StreamingNormalize(
        self._batch_env.observ[0], center=True, scale=True, clip=5,
        name='normalize_observ')
    self._reward_filter = parts.StreamingNormalize(
        self._batch_env.reward[0], center=False, scale=True, clip=10,
        name='normalize_reward')
    self._use_gpu = self._config.use_gpu and utility.available_gpus()
    policy_params, state = self._initialize_policy()
    self._initialize_memory(policy_params)
    # Initialize the optimizer and penalty.
    with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
      self._optimizer = self._config.optimizer(self._config.learning_rate)
    self._penalty = tf.Variable(
        self._config.kl_init_penalty, False, dtype=tf.float32)
    # If the policy is stateful, allocate space to store its state.
    with tf.variable_scope('ppo_temporary'):
      with tf.device('/gpu:0'):
        if state is None:
          self._last_state = None
        else:
          var_like = lambda x: tf.Variable(lambda: tf.zeros_like(x), False)
          self._last_state = tools.nested.map(var_like, state)
    # Remember the action and policy parameters to write into the memory.
    with tf.variable_scope('ppo_temporary'):
      self._last_action = tf.Variable(
          tf.zeros_like(self._batch_env.action), False, name='last_action')
      self._last_policy = tools.nested.map(
          lambda x: tf.Variable(tf.zeros_like(x[:, 0], False)), policy_params)

  def begin_episode(self, agent_indices):
    """Reset the recurrent states and stored episode.

    Args:
      agent_indices: Tensor containing current batch indices.

    Returns:
      Summary tensor.
    """
    with tf.name_scope('begin_episode/'):
      if self._last_state is None:
        reset_state = tf.no_op()
      else:
        reset_state = utility.reinit_nested_vars(
            self._last_state, agent_indices)
      reset_buffer = self._current_episodes.clear(agent_indices)
      with tf.control_dependencies([reset_state, reset_buffer]):
        return tf.constant('')

  def perform(self, agent_indices, observ):
    """Compute batch of actions and a summary for a batch of observation.

    Args:
      agent_indices: Tensor containing current batch indices.
      observ: Tensor of a batch of observations for all agents.

    Returns:
      Tuple of action batch tensor and summary tensor.
    """
    with tf.name_scope('perform/'):
      observ = self._observ_filter.transform(observ)
      if self._last_state is None:
        state = None
      else:
        state = tools.nested.map(
            lambda x: tf.gather(x, agent_indices), self._last_state)
      with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
        output = self._network(
            observ[:, None], tf.ones(observ.shape[0]), state)
      action = tf.cond(
          self._is_training, output.policy.sample, output.policy.mode)
      logprob = output.policy.log_prob(action)[:, 0]
      # pylint: disable=g-long-lambda
      summary = tf.cond(self._should_log, lambda: tf.summary.merge([
          tf.summary.histogram('mode', output.policy.mode()[:, 0]),
          tf.summary.histogram('action', action[:, 0]),
          tf.summary.histogram('logprob', logprob)]), str)
      # Remember current policy to append to memory in the experience callback.
      if self._last_state is None:
        assign_state = tf.no_op()
      else:
        assign_state = utility.assign_nested_vars(
            self._last_state, output.state, agent_indices)
      remember_last_action = tf.scatter_update(
          self._last_action, agent_indices, action[:, 0])
      policy_params = tools.nested.filter(
          lambda x: isinstance(x, tf.Tensor), output.policy.parameters)
      assert policy_params, 'Policy has no parameters to store.'
      remember_last_policy = tools.nested.map(
          lambda var, val: tf.scatter_update(var, agent_indices, val[:, 0]),
          self._last_policy, policy_params, flatten=True)
      with tf.control_dependencies((
          assign_state, remember_last_action) + remember_last_policy):
        return action[:, 0], tf.identity(summary)

  def experience(
      self, agent_indices, observ, action, reward, unused_done, unused_nextob):
    """Process the transition tuple of the current step.

    When training, add the current transition tuple to the memory and update
    the streaming statistics for observations and rewards. A summary string is
    returned if requested at this step.

    Args:
      agent_indices: Tensor containing current batch indices.
      observ: Batch tensor of observations.
      action: Batch tensor of actions.
      reward: Batch tensor of rewards.
      unused_done: Batch tensor of done flags.
      unused_nextob: Batch tensor of successor observations.

    Returns:
      Summary tensor.
    """
    with tf.name_scope('experience/'):
      return tf.cond(
          self._is_training,
          # pylint: disable=g-long-lambda
          lambda: self._define_experience(
              agent_indices, observ, action, reward), str)

  def _define_experience(self, agent_indices, observ, action, reward):
    """Implement the branch of experience() entered during training."""
    update_filters = tf.summary.merge([
        self._observ_filter.update(observ),
        self._reward_filter.update(reward)])
    with tf.control_dependencies([update_filters]):
      if self._config.train_on_agent_action:
        # NOTE: Doesn't seem to change much.
        action = self._last_action
      policy = tools.nested.map(
          lambda x: tf.gather(x, agent_indices), self._last_policy)
      batch = (observ, action, policy, reward)
      append = self._current_episodes.append(batch, agent_indices)
    with tf.control_dependencies([append]):
      norm_observ = self._observ_filter.transform(observ)
      norm_reward = tf.reduce_mean(self._reward_filter.transform(reward))
      # pylint: disable=g-long-lambda
      summary = tf.cond(self._should_log, lambda: tf.summary.merge([
          update_filters,
          self._observ_filter.summary(),
          self._reward_filter.summary(),
          tf.summary.scalar('memory_size', self._num_finished_episodes),
          tf.summary.histogram('normalized_observ', norm_observ),
          tf.summary.histogram('action', self._last_action),
          tf.summary.scalar('normalized_reward', norm_reward)]), str)
      return summary

  def end_episode(self, agent_indices):
    """Add episodes to the memory and perform update steps if memory is full.

    During training, add the collected episodes of the batch indices that
    finished their episode to the memory. If the memory is full, train on it,
    and then clear the memory. A summary string is returned if requested at
    this step.

    Args:
      agent_indices: Tensor containing current batch indices.

    Returns:
       Summary tensor.
    """
    with tf.name_scope('end_episode/'):
      return tf.cond(
          self._is_training,
          lambda: self._define_end_episode(agent_indices), str)

  def _initialize_policy(self):
    """Initialize the policy.

    Run the policy network on dummy data to initialize its parameters for later
    reuse and to analyze the policy distribution. Initializes the attributes
    `self._network` and `self._policy_type`.

    Raises:
      ValueError: Invalid policy distribution.

    Returns:
      Parameters of the policy distribution and policy state.
    """
    with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
      network = functools.partial(
          self._config.network, self._config, self._batch_env.action_space)
      self._network = tf.make_template('network', network)
      output = self._network(
          tf.zeros_like(self._batch_env.observ)[:, None],
          tf.ones(len(self._batch_env)))
    if output.policy.event_shape != self._batch_env.action.shape[1:]:
      message = 'Policy event shape {} does not match action shape {}.'
      message = message.format(
          output.policy.event_shape, self._batch_env.action.shape[1:])
      raise ValueError(message)
    self._policy_type = type(output.policy)
    is_tensor = lambda x: isinstance(x, tf.Tensor)
    policy_params = tools.nested.filter(is_tensor, output.policy.parameters)
    set_batch_dim = lambda x: utility.set_dimension(x, 0, len(self._batch_env))
    tools.nested.map(set_batch_dim, policy_params)
    if output.state is not None:
      tools.nested.map(set_batch_dim, output.state)
    return policy_params, output.state

  def _initialize_memory(self, policy_params):
    """Initialize temporary and permanent memory.

    Args:
      policy_params: Nested tuple of policy parameters with all dimensions set.

    Initializes the attributes `self._current_episodes`,
    `self._finished_episodes`, and `self._num_finished_episodes`. The episodes
    memory serves to collect multiple episodes in parallel. Finished episodes
    are copied into the next free slot of the second memory. The memory index
    points to the next free slot.
    """
    # We store observation, action, policy parameters, and reward.
    template = (
        self._batch_env.observ[0],
        self._batch_env.action[0],
        tools.nested.map(lambda x: x[0, 0], policy_params),
        self._batch_env.reward[0])
    with tf.variable_scope('ppo_temporary'):
      self._current_episodes = parts.EpisodeMemory(
          template, len(self._batch_env), self._config.max_length, 'episodes')
    self._finished_episodes = parts.EpisodeMemory(
        template, self._config.update_every, self._config.max_length, 'memory')
    self._num_finished_episodes = tf.Variable(0, False)

  def _define_end_episode(self, agent_indices):
    """Implement the branch of end_episode() entered during training."""
    episodes, length = self._current_episodes.data(agent_indices)
    space_left = self._config.update_every - self._num_finished_episodes
    use_episodes = tf.range(tf.minimum(
        tf.shape(agent_indices)[0], space_left))
    episodes = tools.nested.map(lambda x: tf.gather(x, use_episodes), episodes)
    append = self._finished_episodes.replace(
        episodes, tf.gather(length, use_episodes),
        use_episodes + self._num_finished_episodes)
    with tf.control_dependencies([append]):
      increment_index = self._num_finished_episodes.assign_add(
          tf.shape(use_episodes)[0])
    with tf.control_dependencies([increment_index]):
      memory_full = self._num_finished_episodes >= self._config.update_every
      return tf.cond(memory_full, self._training, str)

  def _training(self):
    """Perform multiple training iterations of both policy and value baseline.

    Training on the episodes collected in the memory. Reset the memory
    afterwards. Always returns a summary string.

    Returns:
      Summary tensor.
    """
    with tf.device('/gpu:0' if self._use_gpu else '/cpu:0'):
      with tf.name_scope('training'):
        assert_full = tf.assert_equal(
            self._num_finished_episodes, self._config.update_every)
        with tf.control_dependencies([assert_full]):
          data = self._finished_episodes.data()
        (observ, action, old_policy_params, reward), length = data
        # We set padding frames of the parameters to ones to prevent Gaussians
        # with zero variance. This would result in an infinite KL divergence,
        # which, even if masked out, would result in NaN gradients.
        old_policy_params = tools.nested.map(
            lambda param: self._mask(param, length, 1), old_policy_params)
        with tf.control_dependencies([tf.assert_greater(length, 0)]):
          length = tf.identity(length)
        observ = self._observ_filter.transform(observ)
        reward = self._reward_filter.transform(reward)
        update_summary = self._perform_update_steps(
            observ, action, old_policy_params, reward, length)
        with tf.control_dependencies([update_summary]):
          penalty_summary = self._adjust_penalty(
              observ, old_policy_params, length)
        with tf.control_dependencies([penalty_summary]):
          clear_memory = tf.group(
              self._finished_episodes.clear(),
              self._num_finished_episodes.assign(0))
        with tf.control_dependencies([clear_memory]):
          weight_summary = utility.variable_summaries(
              tf.trainable_variables(), self._config.weight_summaries)
          return tf.summary.merge([
              update_summary, penalty_summary, weight_summary])

  def _perform_update_steps(
      self, observ, action, old_policy_params, reward, length):
    """Perform multiple update steps of value function and policy.

    The advantage is computed once at the beginning and shared across
    iterations. We need to decide for the summary of one iteration, and thus
    choose the one after half of the iterations.

    Args:
      observ: Sequences of observations.
      action: Sequences of actions.
      old_policy_params: Parameters of the behavioral policy.
      reward: Sequences of rewards.
      length: Batch of sequence lengths.

    Returns:
      Summary tensor.
    """
    return_ = utility.discounted_return(
        reward, length, self._config.discount)
    value = self._network(observ, length).value
    if self._config.gae_lambda:
      advantage = utility.lambda_advantage(
          reward, value, length, self._config.discount,
          self._config.gae_lambda)
    else:
      advantage = return_ - value
    mean, variance = tf.nn.moments(advantage, axes=[0, 1], keep_dims=True)
    advantage = (advantage - mean) / (tf.sqrt(variance) + 1e-8)
    advantage = tf.Print(
        advantage, [tf.reduce_mean(return_), tf.reduce_mean(value)],
        'return and value: ')
    advantage = tf.Print(
        advantage, [tf.reduce_mean(advantage)],
        'normalized advantage: ')
    episodes = (observ, action, old_policy_params, reward, advantage)
    value_loss, policy_loss, summary = parts.iterate_sequences(
        self._update_step, [0., 0., ''], episodes, length,
        self._config.chunk_length,
        self._config.batch_size,
        self._config.update_epochs,
        padding_value=1)
    print_losses = tf.group(
        tf.Print(0, [tf.reduce_mean(value_loss)], 'value loss: '),
        tf.Print(0, [tf.reduce_mean(policy_loss)], 'policy loss: '))
    with tf.control_dependencies([value_loss, policy_loss, print_losses]):
      return summary[self._config.update_epochs // 2]

  def _update_step(self, sequence):
    """Compute the current combined loss and perform a gradient update step.

    The sequences must be a dict containing the keys `length` and `sequence`,
    where the latter is a tuple containing observations, actions, parameters of
    the behavioral policy, rewards, and advantages.

    Args:
      sequence: Sequences of episodes or chunks of episodes.

    Returns:
      Tuple of value loss, policy loss, and summary tensor.
    """
    observ, action, old_policy_params, reward, advantage = sequence['sequence']
    length = sequence['length']
    old_policy = self._policy_type(**old_policy_params)
    value_loss, value_summary = self._value_loss(observ, reward, length)
    network = self._network(observ, length)
    policy_loss, policy_summary = self._policy_loss(
        old_policy, network.policy, action, advantage, length)
    loss = policy_loss + value_loss + network.get('loss', 0)
    gradients, variables = (
        zip(*self._optimizer.compute_gradients(loss)))
    optimize = self._optimizer.apply_gradients(
        zip(gradients, variables))
    summary = tf.summary.merge([
        value_summary, policy_summary,
        tf.summary.histogram('network_loss', network.get('loss', 0)),
        tf.summary.scalar('gradient_norm', tf.global_norm(gradients)),
        utility.gradient_summaries(zip(gradients, variables))])
    with tf.control_dependencies([optimize]):
      return [tf.identity(x) for x in (value_loss, policy_loss, summary)]

  def _value_loss(self, observ, reward, length):
    """Compute the loss function for the value baseline.

    The value loss is the difference between empirical and approximated returns
    over the collected episodes. Returns the loss tensor and a summary strin.

    Args:
      observ: Sequences of observations.
      reward: Sequences of reward.
      length: Batch of sequence lengths.

    Returns:
      Tuple of loss tensor and summary tensor.
    """
    with tf.name_scope('value_loss'):
      value = self._network(observ, length).value
      return_ = utility.discounted_return(
          reward, length, self._config.discount)
      advantage = return_ - value
      value_loss = 0.5 * self._mask(advantage ** 2, length)
      summary = tf.summary.merge([
          tf.summary.histogram('value_loss', value_loss),
          tf.summary.scalar('avg_value_loss', tf.reduce_mean(value_loss))])
      value_loss = tf.reduce_mean(value_loss)
      return tf.check_numerics(value_loss, 'value_loss'), summary

  def _policy_loss(
      self, old_policy, policy, action, advantage, length):
    """Compute the policy loss composed of multiple components.

    1. The policy gradient loss is importance sampled from the data-collecting
       policy at the beginning of training.
    2. The second term is a KL penalty between the policy at the beginning of
       training and the current policy.
    3. Additionally, if this KL already changed more than twice the target
       amount, we activate a strong penalty discouraging further divergence.

    Args:
      old_policy: Action distribution of the behavioral policy.
      policy: Sequences of distribution params of the current policy.
      action: Sequences of actions.
      advantage: Sequences of advantages.
      length: Batch of sequence lengths.

    Returns:
      Tuple of loss tensor and summary tensor.
    """
    with tf.name_scope('policy_loss'):
      kl = tf.contrib.distributions.kl_divergence(old_policy, policy)
      # Infinite values in the KL, even for padding frames that we mask out,
      # cause NaN gradients since TensorFlow computes gradients with respect to
      # the whole input tensor.
      kl = tf.check_numerics(kl, 'kl')
      kl = tf.reduce_mean(self._mask(kl, length), 1)
      policy_gradient = tf.exp(
          policy.log_prob(action) - old_policy.log_prob(action))
      surrogate_loss = -tf.reduce_mean(self._mask(
          policy_gradient * tf.stop_gradient(advantage), length), 1)
      surrogate_loss = tf.check_numerics(surrogate_loss, 'surrogate_loss')
      kl_penalty = self._penalty * kl
      cutoff_threshold = self._config.kl_target * self._config.kl_cutoff_factor
      cutoff_count = tf.reduce_sum(
          tf.cast(kl > cutoff_threshold, tf.int32))
      with tf.control_dependencies([tf.cond(
          cutoff_count > 0,
          lambda: tf.Print(0, [cutoff_count], 'kl cutoff! '), int)]):
        kl_cutoff = (
            self._config.kl_cutoff_coef *
            tf.cast(kl > cutoff_threshold, tf.float32) *
            (kl - cutoff_threshold) ** 2)
      policy_loss = surrogate_loss + kl_penalty + kl_cutoff
      entropy = tf.reduce_mean(policy.entropy(), axis=1)
      if self._config.entropy_regularization:
        policy_loss -= self._config.entropy_regularization * entropy
      summary = tf.summary.merge([
          tf.summary.histogram('entropy', entropy),
          tf.summary.histogram('kl', kl),
          tf.summary.histogram('surrogate_loss', surrogate_loss),
          tf.summary.histogram('kl_penalty', kl_penalty),
          tf.summary.histogram('kl_cutoff', kl_cutoff),
          tf.summary.histogram('kl_penalty_combined', kl_penalty + kl_cutoff),
          tf.summary.histogram('policy_loss', policy_loss),
          tf.summary.scalar('avg_surr_loss', tf.reduce_mean(surrogate_loss)),
          tf.summary.scalar('avg_kl_penalty', tf.reduce_mean(kl_penalty)),
          tf.summary.scalar('avg_policy_loss', tf.reduce_mean(policy_loss))])
      policy_loss = tf.reduce_mean(policy_loss, 0)
      return tf.check_numerics(policy_loss, 'policy_loss'), summary

  def _adjust_penalty(self, observ, old_policy_params, length):
    """Adjust the KL policy between the behavioral and current policy.

    Compute how much the policy actually changed during the multiple
    update steps. Adjust the penalty strength for the next training phase if we
    overshot or undershot the target divergence too much.

    Args:
      observ: Sequences of observations.
      old_policy_params: Parameters of the behavioral policy.
      length: Batch of sequence lengths.

    Returns:
      Summary tensor.
    """
    old_policy = self._policy_type(**old_policy_params)
    with tf.name_scope('adjust_penalty'):
      network = self._network(observ, length)
      print_penalty = tf.Print(0, [self._penalty], 'current penalty: ')
      with tf.control_dependencies([print_penalty]):
        kl_change = tf.reduce_mean(self._mask(
            tf.contrib.distributions.kl_divergence(old_policy, network.policy),
            length))
        kl_change = tf.Print(kl_change, [kl_change], 'kl change: ')
        maybe_increase = tf.cond(
            kl_change > 1.3 * self._config.kl_target,
            # pylint: disable=g-long-lambda
            lambda: tf.Print(self._penalty.assign(
                self._penalty * 1.5), [0], 'increase penalty '),
            float)
        maybe_decrease = tf.cond(
            kl_change < 0.7 * self._config.kl_target,
            # pylint: disable=g-long-lambda
            lambda: tf.Print(self._penalty.assign(
                self._penalty / 1.5), [0], 'decrease penalty '),
            float)
      with tf.control_dependencies([maybe_increase, maybe_decrease]):
        return tf.summary.merge([
            tf.summary.scalar('kl_change', kl_change),
            tf.summary.scalar('penalty', self._penalty)])

  def _mask(self, tensor, length, padding_value=0):
    """Set padding elements of a batch of sequences to a constant.

    Useful for setting padding elements to zero before summing along the time
    dimension, or for preventing infinite results in padding elements.

    Args:
      tensor: Tensor of sequences.
      length: Batch of sequence lengths.
      padding_value: Value to write into padding elements.

    Returns:
      Masked sequences.
    """
    with tf.name_scope('mask'):
      range_ = tf.range(tensor.shape[1].value)
      mask = range_[None, :] < length[:, None]
      if tensor.shape.ndims > 2:
        for _ in range(tensor.shape.ndims - 2):
          mask = mask[..., None]
        mask = tf.tile(mask, [1, 1] + tensor.shape[2:].as_list())
      masked = tf.where(mask, tensor, padding_value * tf.ones_like(tensor))
      return tf.check_numerics(masked, 'masked')
