#!/usr/bin/env python3

import os
import time

import datetime
import numpy as np
import tensorflow as tf

# Load the episode length
ep_length = int(open("ep_length.txt", 'rb').read().decode(encoding="utf-8"))

# Load the dataset and decode it to txt
text = open("dataset.txt", 'rb').read().decode(encoding="utf-8")

print("Text Length:\t{}".format(len(text)))

# Get all unique characters in a list
vocab = sorted(set(text))

print("Unique Characters:\t{}".format(len(vocab)))

# Correlate the vocab list to an int (char:int)
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Load the dataset into a list in int form based of the previously made list
text_as_int = np.array([char2idx[c] for c in text])

# Set the sequence length and examples per epoch
seq_length = 800
examples_per_epoch = len(text) // seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# Create batches
BATCH_SIZE = 24

BUFFER_SIZE = 5000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)

embedding_dim = 512

rnn_units = 1024


# Build model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# Load previous checkpoint if needed
response = input("Load load weights from checkpoint? (y/n): ").lower()

if response == 'y':
  model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
else:
  pass

# Set how many epochs to complete
try:
  EPOCHS = int(input("How many epochs should I do?"))
except:
  print("You fucked up somehow so I'll only do 1")
  EPOCHS = 1

# Training time!
while True:
  history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
  print("Keep Training?")
  response = input("(Y/N): ").lower()

  if response == 'y':
    try:
      EPOCHS = int(input("How many epochs should I do?"))
    except:
      print("You fucked up somehow so I'll only do 1")
      EPOCHS = 1
    continue
  elif response == 'n':
    break
  else:
    print("Invalid response, training another epoch anyways")
    continue


# Build the model
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

# Generate the text from model output (based on start_string)
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 2000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


# Display the generated text
print(generate_text(model, start_string="YLYL 0000\n"))
