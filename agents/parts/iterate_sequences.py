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

"""Normalize tensors based on streaming estimates of mean and variance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from agents import tools


def iterate_sequences(
    consumer_fn, output_template, sequences, length, chunk_length=None,
    batch_size=None, num_epochs=1, padding_value=0):
  """Iterate over batches of chunks of sequences for multiple epochs.

  The batch dimension of the length tensor must be set because it is used to
  infer buffer sizes.

  Args:
    consumer_fn: Function creating the operation to process the data.
    output_template: Nested tensors of same shape and dtype as outputs.
    sequences: Nested collection of tensors with batch and time dimension.
    length: Tensor containing the length for each sequence.
    chunk_length: Split sequences into chunks of this size; optional.
    batch_size: Split epochs into batches of this size; optional.
    num_epochs: How many times to repeat over the data.
    padding_value: Value used for padding the last chunk after the sequence.

  Raises:
    ValueError: Unknown batch size of the length tensor.

  Returns:
    Concatenated nested tensors returned by the consumer.
  """
  if not length.shape[0].value:
    raise ValueError('Batch size of length tensor must be set.')
  num_sequences = length.shape[0].value
  sequences = dict(sequence=sequences, length=length)
  dataset = tf.data.Dataset.from_tensor_slices(sequences)
  dataset = dataset.repeat(num_epochs)
  if chunk_length:
    dataset = dataset.map(remove_padding).flat_map(
        # pylint: disable=g-long-lambda
        lambda x: tf.data.Dataset.from_tensor_slices(
            chunk_sequence(x, chunk_length, padding_value)))
    num_chunks = tf.reduce_sum((length - 1) // chunk_length + 1)
  else:
    num_chunks = num_sequences
  if batch_size:
    dataset = dataset.shuffle(num_sequences // 2)
  dataset = dataset.batch(batch_size or num_sequences)
  dataset = dataset.prefetch(num_epochs)
  iterator = dataset.make_initializable_iterator()
  with tf.control_dependencies([iterator.initializer]):
    num_batches = num_epochs * num_chunks // (batch_size or num_sequences)
    return tf.scan(
        # pylint: disable=g-long-lambda
        lambda _1, index: consumer_fn(iterator.get_next()),
        tf.range(num_batches), output_template, parallel_iterations=1)


def chunk_sequence(sequence, chunk_length=200, padding_value=0):
  """Split a nested dict of sequence tensors into a batch of chunks.

  This function does not expect a batch of sequences, but a single sequence. A
  `length` key is added if it did not exist already.

  Args:
    sequence: Nested dict of tensors with time dimension.
    chunk_length: Size of chunks the sequence will be split into.
    padding_value: Value used for padding the last chunk after the sequence.

  Returns:
    Nested dict of sequence tensors with chunk dimension.
  """
  if 'length' in sequence:
    length = sequence.pop('length')
  else:
    length = tf.shape(tools.nested.flatten(sequence)[0])[0]
  num_chunks = (length - 1) // chunk_length + 1
  padding_length = chunk_length * num_chunks - length
  padded = tools.nested.map(
      # pylint: disable=g-long-lambda
      lambda tensor: tf.concat([
          tensor, 0 * tensor[:padding_length] + padding_value], 0),
      sequence)
  chunks = tools.nested.map(
      # pylint: disable=g-long-lambda
      lambda tensor: tf.reshape(
          tensor, [num_chunks, chunk_length] + tensor.shape[1:].as_list()),
      padded)
  chunks['length'] = tf.concat([
      chunk_length * tf.ones((num_chunks - 1,), dtype=tf.int32),
      [chunk_length - padding_length]], 0)
  return chunks


def remove_padding(sequence):
  """Selects the used frames of a sequence, up to its length.

  This function does not expect a batch of sequences, but a single sequence.
  The sequence must be a dict with `length` key, which will removed from the
  result.

  Args:
    sequence: Nested dict of tensors with time dimension.

  Returns:
    Nested dict of tensors with padding elements and `length` key removed.
  """
  length = sequence.pop('length')
  sequence = tools.nested.map(lambda tensor: tensor[:length], sequence)
  return sequence
