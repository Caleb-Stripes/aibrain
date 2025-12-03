"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import sys
import numpy as np
import json
import os

# data I/O
data = open('input.txt', 'r').read()  # should be simple plain text file
# IMPORTANT: use deterministic ordering of characters to ensure consistent indexing across runs.
# Using list(set(data)) leads to arbitrary ordering due to Python hash randomization; breaks loaded weights.
initial_chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(initial_chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(initial_chars)}
ix_to_char = {i: ch for i, ch in enumerate(initial_chars)}

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

def save_model(filename='rnn_model.json', hprev_state=None):
  """Save model parameters to a JSON file"""
  model_params = {
    'Wxh': Wxh.tolist(),
    'Whh': Whh.tolist(),
    'Why': Why.tolist(),
    'bh': bh.tolist(),
    'by': by.tolist(),
    'mWxh': mWxh.tolist(),
    'mWhh': mWhh.tolist(),
    'mWhy': mWhy.tolist(),
    'mbh': mbh.tolist(),
    'mby': mby.tolist(),
    'hprev': hprev_state.tolist() if hprev_state is not None else None,
    'chars': [ix_to_char[i] for i in range(len(ix_to_char))],  # preserve vocabulary ordering
    'n': n,
    'p': p,
    'smooth_loss': smooth_loss,
    'hidden_size': hidden_size,
    'vocab_size': vocab_size
  }
  with open(filename, 'w') as f:
    json.dump(model_params, f, indent=4)
  print(f'Model saved to {filename}')

def load_model(filename='rnn_model.json'):
  """Load model parameters from a JSON file"""
  if not os.path.exists(filename):
    return None
  
  with open(filename, 'r') as f:
    model_params = json.load(f)
  
  print(f'Model loaded from {filename}')
  return {
    'weights': (
      np.array(model_params['Wxh']),
      np.array(model_params['Whh']),
      np.array(model_params['Why']),
      np.array(model_params['bh']),
      np.array(model_params['by'])
    ),
    'memory': (
      np.array(model_params.get('mWxh', [])),
      np.array(model_params.get('mWhh', [])),
      np.array(model_params.get('mWhy', [])),
      np.array(model_params.get('mbh', [])),
      np.array(model_params.get('mby', []))
    ) if 'mWxh' in model_params else None,
    'hprev': np.array(model_params['hprev']) if model_params.get('hprev') is not None else None,
    'chars': model_params.get('chars'),
    'training_state': {
      'n': model_params.get('n', 0),
      'p': model_params.get('p', 0),
      'smooth_loss': model_params.get('smooth_loss', -np.log(1.0/model_params['vocab_size'])*seq_length)
    }
  }

# model parameters
# Try to load existing model, otherwise initialize randomly
loaded_data = load_model()
if loaded_data is not None:
  # Rebuild vocabulary mappings from saved chars to ensure index consistency.
  saved_chars = loaded_data.get('chars')
  if saved_chars is None:
    # Fallback: use deterministic initial_chars but warn.
    print('Warning: Saved model missing chars list; using current sorted chars. Index mismatch may occur.')
    saved_chars = initial_chars
  char_to_ix = {ch: i for i, ch in enumerate(saved_chars)}
  ix_to_char = {i: ch for i, ch in enumerate(saved_chars)}
  # Update vocab_size in case file changed
  vocab_size = len(saved_chars)
  Wxh, Whh, Why, bh, by = loaded_data['weights']
  if loaded_data['memory'] is not None and len(loaded_data['memory'][0]) > 0:
    mWxh, mWhh, mWhy, mbh, mby = loaded_data['memory']
  else:
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
  n = loaded_data['training_state']['n']
  p = loaded_data['training_state']['p']
  smooth_loss = loaded_data['training_state']['smooth_loss']
  hprev = loaded_data['hprev'] if loaded_data['hprev'] is not None else np.zeros((hidden_size, 1))
  print(f'Using loaded model parameters (iteration {n}, loss {smooth_loss:.2f})')
else:
  Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
  Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
  Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
  bh = np.zeros((hidden_size, 1))  # hidden bias
  by = np.zeros((vocab_size, 1))  # output bias
  mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
  n, p = 0, 0
  smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
  hprev = np.zeros((hidden_size, 1))
  print('Using randomly initialized parameters')
  
def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes


# TODO: modify the training loop to conclude when achieving a certain loss threshold & then save the model parameters to a file.

while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p + seq_length + 1 >= len(data):
    p = 0  # wrap to start of data
    # Do NOT reset hprev here; allow hidden continuity across epochs.
  if n == 0:
    hprev = np.zeros((hidden_size, 1))  # only reset at very beginning of training
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))


  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # save model every 1000 iterations
  if n % 1000 == 0 and n > 0:
    save_model(hprev_state=hprev)
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
  # sys.exit(0)
  