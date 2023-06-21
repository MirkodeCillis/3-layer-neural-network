import numpy as np
import scipy.special

# neural network class definition
class NeuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # learning rate
        self.lr = learningrate
        # link weights matrix between input and hidden layers (wih) and
        # between hidden and output layers are set. Indexes w[i, j] represent
        # respectively the previous and the next layer nodes.
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # set the activation function
        self.actFunct = lambda x: scipy.special.expit(x)
        self.invFunct = lambda x: scipy.special.logit(x)
        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        # monodimensional matrix [1xn]. The 'T' attribute
        # represents the transpose of the matrix [nx1].

        # compute signals into the hidden layer
        hiddenIn = np.dot(self.wih, inputs)
        # compute output signals from the hidden layer
        hiddenOut = self.actFunct(hiddenIn)
        # the same process is done for the output layer
        finalIn = np.dot(self.who, hiddenOut)
        finalOut = self.actFunct(finalIn)
        return finalOut

    # train the neural network
    def train(self, inputs_list, targets_list):
        # PART 1: compute the output. it consists of
        # the same operations as the method query().
        inputs = np.array(inputs_list, ndmin=2).T
        hiddenIn = np.dot(self.wih, inputs)
        hiddenOut = self.actFunct(hiddenIn)
        finalIn = np.dot(self.who, hiddenOut)
        finalOut = self.actFunct(finalIn)
        # PART 2: generate the error and backpropagate it.
        targets = np.array(targets_list, ndmin=2).T
        # the output layer error matrix will be used to refine
        # the link weights among the hidden and output layers.
        outputErr = targets - finalOut
        # the hidden layer error matrix will be used to refine
        # the link weights among the input and hidden layers.
        hiddenErr = np.dot(self.who.T, outputErr)
        # refining the link weights among the input and hidden layers
        self.wih += self.lr * np.dot((hiddenErr * hiddenOut * (1.0 - hiddenOut)), inputs.T)
        # refining the link weights among the hidden and output layers
        self.who += self.lr * np.dot((outputErr * finalOut * (1.0 - finalOut)), hiddenOut.T)
        pass

    # Backward Query
    def backquery(self, output_list):
        # convert inputs list to 2D array
        final_outputs = np.array(output_list, ndmin=2).T
        # calculate the signal into the final output layer
        final_inputs = self.invFunct(final_outputs)
        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        # calculate the signal into the hidden layer
        hidden_inputs = self.invFunct(hidden_outputs)
        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        return inputs

    # import/export of weights so that they won't be forgotten as the execution ends
    def importWeights(self, fpath):
        try:
            with open(fpath, "r") as f:
                weights = f.readline().split(", ")
                weights.pop()  # newline must be removed
                # the list must be set as a matrix with the correct dimension
                # for example, the first list must be reshaped according to the
                # dimension of the input and hidden layers.
                self.wih = np.asfarray(weights).reshape((self.hnodes, self.inodes))
                weights = f.readline().split(", ")
                weights.pop()
                self.who = np.asfarray(weights).reshape((self.onodes, self.hnodes))
                return True
        except:  # in case of an error, the import is canceled
            return False
        pass

    def exportWeights(self, fpath):
        with open(fpath, "w") as f:
            for i in range(self.hnodes):  # self.hnodes rows
                for h in range(self.inodes):  # self.inodes columns
                    f.write(str(self.wih[i, h]) + ", ")
                    pass
                pass
            # weights of different links are separated by a newline
            f.write('\n')
            for h in range(self.onodes):
                for o in range(self.hnodes):
                    f.write(str(self.who[h, o]) + ", ")
                    pass
                pass
            pass
        pass

    pass
