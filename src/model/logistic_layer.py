import numpy as np

from util.activation_functions import Activation

class LogisticLayer():
    """
    A layer of neural

    Parameters
    ----------
    n_in: int: number of units from the previous layer (or input data)
    n_out: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    is_classifier_layer: bool:  to do classification or regression

    Attributes
    ----------
    n_in : positive int:
        number of units from the previous layer
    n_out : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activation_string : string
        the name of the activation function
    is_classifier_layer: bool
        to do classification or regression
    deltas : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, n_in, n_out, weights=None,
                 activation='sigmoid', is_classifier_layer=False):

        # Get activation function from string
        self.activation_string = activation
        self.activation = Activation.get_activation(self.activation_string)

        self.n_in = n_in
        self.n_out = n_out

        self.outp = np.ndarray((n_out,))
        self.deltas = np.zeros((n_out,))
        self.inp = np.ndarray((n_in+1,))
        self.inp[0] = 1 # bias

        # You can have better initialization here
        if weights is None:
            # self.weights = np.random.rand(n_out, n_in + 1)/10
            self.weights = np.random.rand(n_in + 1, n_out)/10 # +1 added by me for bias
        else:
            self.weights = weights

        self.is_classifier_layer = is_classifier_layer

        # Some handy properties of the layers
        self.size = self.n_out
        self.shape = self.weights.shape

    def forward(self, inp):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        inp : ndarray
            a numpy ndarray of size (n_in,) containing the input of the layer

        Change outp
        -------
        outp: array
            a numpy ndarray of size (n_out,) containing the output of the layer
        """

        # Here you have to implement the forward pass
        # fire each neuron
        self.inp[1:] = inp  # keep bias
        self.outp = self._fire(self.inp)

    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (backward)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer
            ATTENTION: WEIGHTS FOR BIAS INCLUDED

        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """
        # fully connected, skip bias weights
        self.deltas = self.outp * (1.0 - self.outp) * map(sum, nextDerivatives * nextWeights[1:,])

    def updateWeights(self, learningRate):
        """
        Update the weights of the layer
        """

        # Here the implementation of weight updating mechanism
        for i in range(self.n_in + 1):
            self.weights[i,] += learningRate * self.deltas * self.inp[i]

    def _fire(self, inp):
        return self.activation(np.dot(np.array(inp), self.weights))
