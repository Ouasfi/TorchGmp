import torch 
import numpy as np
from copy import deepcopy
from integer_ops import IntegerTensor

def einsum_ij(a, b):
    batch = a.size(0)
    n_a = a.size(1)
    n_b = b.size(1)
    return (a.repeat(n_b, 1,1).permute(1,2,0).flatten() * b.repeat(n_a,1,1).permute(1,0,2).flatten()).reshape(batch, n_a,n_b)
class IntegerMLP(object):
    def __init__(self, layers=[2, 10, 1], activations=['sigmoid', 'sigmoid'], t = 5):
        """The list ``layers`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and std sqrt(2/number_of_neurones_at_each_layer).  Note that the first
        layer is assumed to be an input layer"""
        assert (len(layers) == len(activations) + 1)
        self.layers = layers
        self.activations = activations
        self.t = t
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            self.weights.append(
                IntegerTensor(torch.randn( size=(layers[i + 1], layers[i]))*q_factor*np.sqrt(2 / layers[i])
                ,q_factor).long())
            self.biases.append(
                IntegerTensor(torch.randn( size=(layers[i + 1], 1))*q_factor* np.sqrt(2 / layers[i])
                ,q_factor).long())
    def encrypt(self):
        self.weights = [w.encrypt() for w in self.weights]
        self.biases = [w.encrypt() for w in self.biases]
    def decrypt(self):
        self.weights = [w.decrypt() for w in self.weights]
        self.biases = [w.decrypt() for w in self.biases]
    def feedforward(self, x):
        """ return the feedforward value for x """
        a = x.clone()

        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            activation_function = self.get_activation_function(self.activations[i])
            #print()
            #print(a.shape, self.weights[i].shape, self.biases[i].shape)
            #print()
            z_s.append(a.matmul( self.weights[i].t())  + self.biases[i].long().squeeze(-1))
            a = activation_function(z_s[-1])
            #print()
            #print(z_s[-1].shape)
            #print()
            a_s.append(a)
        return z_s, a_s

    def train(self, training_data, labels, batch_size=10, epochs=100, lr=0.01, test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` and labels are lists
         representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        """
        if test_data:
            n_test = len(test_data)
        training_data_size = len(training_data)
        for j in range(epochs):
            mini_batches = [training_data[k:k+batch_size] for k in range(0, training_data_size, batch_size)]
            mini_labels = [labels[k:k+batch_size] for k in range(0, training_data_size, batch_size)]
            
            for i, (mini_batch, mini_label) in enumerate(zip(mini_batches, mini_labels)):
                self.update_mini_batch(mini_batch, mini_label, lr)
            
            if j%self.t==0:
                self.get_accuracy(training_data, labels, q_factor)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, mini_label, lr):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch and mini labels.
        The ``mini_batch` and ''mini_label'' are lists of ``(x, train_labels)``, and ``lr``
        is the learning rate.
        """
        #print("Step")
        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch, mini_label)
        self.weights = [w +  torch.div(dweight.sum(0), lr *len(mini_batch), rounding_mode ='floor')
                        for w, dweight in zip(self.weights, delta_nabla_w)]
        self.biases = [w.squeeze(-1) + torch.div(dbias.squeeze(-1).sum(0) , lr*len(mini_batch), rounding_mode ='floor') 
                       for w, dbias in zip(self.biases, delta_nabla_b)]
        
    def backprop(self, input, target):
        """
        Return a tuple ``(db, dw)`` representing the
        gradient for the cost function.  ``db`` and
        ``dw`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.
        """
        #print("Backprop")
        z_s, a_s = self.feedforward(input)
        deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer
        # insert the last layer error
      
        deltas[-1] = ((target - a_s[-1])  * (self.get_derivative_activation_function(self.activations[-1]))(z_s[-1]))
        # Perform BackPropagation
        for i in reversed(range(len(deltas) - 1)):
            deltas[i] =  deltas[i + 1].matmul( self.weights[i + 1],) * (
                self.get_derivative_activation_function(self.activations[i])(z_s[i])) 
        db = deepcopy(deltas)  # dC/dW
        
        #torch.einsum('bi,bj->bij',  deltas[i], a_s[i]) 
        dw = [ einsum_ij(deltas[i],a_s[i] )  for i, d in enumerate(deltas)]  # dC/dB
        # return the derivatives respect to weight matrix and biases
        
        return db, dw

    @staticmethod
    def get_activation_function(name):
        """
        :param name: the name of the activation function (for this work, we implement only sigmoid, relu and linear)
        :return: the activation functions
        """
        if name == 'sigmoid':
            return lambda x: np.exp(x) / (1 + np.exp(x))
        elif name == 'linear':
            return lambda x: x
        elif name == 'relu':
            def relu(x):
                y = x.clone()
                y = y.relu()
                return y
            return relu
        else:
            print('Unknown activation function. linear is used')
            return lambda x: x

    @staticmethod
    def get_derivative_activation_function(name):
        """
        :param name:
        :return:derivative of the activation functions
        """

        if name == 'linear':
            return lambda x: IntegerTensor(torch.tensor([q_factor]), q_factor)
        elif name == 'relu':
            def relu_diff(x):
                y = x.clone()
                y[y >= 0] = 1 
                y[y < 0] = 0
                return y
            return relu_diff
        else:
            print('Unknown activation function. linear is used')
            return lambda x: 1

    def get_accuracy(self, X, Y, q):
        """

        :param X: data
        :param Y: labels
        :param q: an optional parameter for learning on float data
        :return: print the accuracy of the model
        """
        print("Acc")        
        _, a_ = self.feedforward(X)
        acc = 1- (Y.argmax(axis = 1)-a_[-1].argmax(axis = 1)).abs().float().mean()
        print("accuracy == ", acc)




if __name__ == "__main__":
 
    db_size = 1000; q_factor = 2 ** 10
    X_1 = np.random.randint(2, size=db_size)
    X_2 = np.random.randint(2, size=db_size)
  
    q = 1
    # Generate train data and labels
    train_data = q_factor *np.array([(np.array([q * x_1, q * x_2]).reshape(2, 1)) for x_1, x_2 in zip(X_1, X_2)])

    train_labels = q_factor * np.array([np.array([int(np.logical_xor(x_1, x_2)), 1 -
                                            int(np.logical_xor(x_1, x_2))]).reshape(2, 1)
                                            for x_1, x_2 in zip(X_1, X_2)])

    # Define architecture of the MLP
    nn = IntegerMLP([2, 10, 2], activations=['relu', 'linear'], t = 1)
    #nn.encrypt()
    train_data = IntegerTensor(torch.tensor(train_data).squeeze(-1).long(), q_factor)
    train_labels = IntegerTensor(torch.tensor(train_labels).squeeze(-1).long(), q_factor)
    nn.train(train_data, train_labels,
            epochs=20,
            batch_size=40, lr=50
            
    )