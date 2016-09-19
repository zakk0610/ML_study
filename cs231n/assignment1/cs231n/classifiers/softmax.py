import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #print W.shape #(3073, 10)
  #print X.shape #(500, 3073)
  #print y.shape #(500,)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
        scores = X[i].dot(W)
        scores = scores - np.max(scores)  # Numeric stability.
        exp_scores = np.exp(scores)
        #print prob
        loss += -scores[y[i]] + np.log(np.sum(exp_scores))
        for j in xrange(num_classes):
            dW[:, j] += exp_scores[j] / np.sum(exp_scores) * X[i,:]
            if j == y[i]:
                dW[:, j] += -X[i,:]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #print W.shape #(3073, 10)
  #print X.shape #(500, 3073)
  #print y.shape #(500,)

  num_train = X.shape[0]
  scores = np.dot(X, W)  # 500 x 10

  max_val = np.max(scores, axis=1, keepdims=True) # 500
  scores = scores - max_val  # Numeric stability.
  exp_scores = np.exp(scores)
  # loss_i = -f_yi + log(summation of e^fj)
  loss = np.sum(-1 * scores[range(num_train), y]) + np.sum(np.log(np.sum(exp_scores, axis=1)))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  # grads
  # tmp = (e^Wjxj / (summation all of e^fxj),  i is index of training data
  #dw[Wyi] = -xi + tmp x xi = (-1 + tmp) x xi
  #dw[Wyj] = tmp x xi

  tmp = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  tmp[range(num_train), y[range(num_train)]] += -1
  dW = np.dot(X.T, tmp)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

