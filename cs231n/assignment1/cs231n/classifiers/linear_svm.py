import numpy as np
from random import shuffle

# reference code from http://cs231n.github.io/optimization-1/
def svm_loss(W, X, y, reg):
   # compute the loss and the gradient
   num_classes = W.shape[1]
   num_train = X.shape[0]
   loss = 0.0
   for i in xrange(num_train):
     scores = X[i].dot(W)
     correct_class_score = scores[y[i]]
     for j in xrange(num_classes):
       if j == y[i]:
         continue
       margin = scores[j] - correct_class_score + 1 # note delta = 1
       if margin > 0:
         loss += margin
   # Right now the loss is a sum over all training examples, but we want it
   # to be an average instead so we divide by num_train.
   loss /= num_train
   # Add regularization to the loss.
   loss += 0.5 * reg * np.sum(W * W)
   return loss

# same above
def eval_numerical_gradient(f, x):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  #fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x (3073)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index    # index (0,0),(0,1)...
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = old_value - h
    fxmh = f(x)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxph - fxmh) / 2*h # the slope
    it.iternext() # step to next dimension
  return grad

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #print 'W.shape', W.shape # 3073 x 10
  #print 'X.shape', X.shape # 500 x 3073
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        count+=1
        loss += margin
        dW[:, j] += X[i,:]
    dW[:, y[i]] -= count * X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_train = y.shape
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  #'W.shape', W.shape # 3073 x 10
  #'X.shape', X.shape # 500 x 3073
  # y.shape # 500

  scores =  X.dot(W)  # 500 x 10
  index = np.arange(scores.shape[0]).astype(np.int)
  scores_y = np.array([scores[[index,y]]]).transpose()
  margins = np.maximum(0, scores - scores_y +1.0)
  margins[index,y] = 0
  #print scores[0]
  #print y[0]
  #print margins[0]
  
  loss = np.sum(margins)
  loss /= num_train
  # Add regularization to the loss.
  loss += (0.5 * reg) * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dscores = np.zeros(scores.shape) #500x10
  dscores[margins > 0] = 1
  sum_train = np.sum(margins > 0, axis=1) # sum of each train
  dscores[index, y] = -sum_train
  dW = X.T.dot(dscores)/num_train + reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
