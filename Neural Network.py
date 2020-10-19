import numpy as np
from scipy.stats import truncnorm

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
activation = sigmoid

#def relu(x):
    #if x.all()<0:
   #     return 0
  #  else:
 #       return x.all()
#activation = relu

class ANN: 
    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs, no_outputs, no_hidden_layers, 
                 learning_rate, bias):

        self.no_inputs = no_inputs
        self.no_outputs = no_outputs
        self.no_hidden_layers = no_hidden_layers
        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weights()

        # TODO initialise weights
    def create_weights(self):
        bias_node = 1 if self.bias else 0
        rad = 1 / np.sqrt(self.no_inputs + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.hidden_weights_in = X.rvs((self.no_hidden_layers, self.no_inputs + bias_node))
        rad = 1 / np.sqrt(self.no_hidden_layers + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.hidden_weights_out = X.rvs((self.no_outputs, self.no_hidden_layers + bias_node))
        
        
    #===============================#
    # Trains the net using labelled #
    # training data.                #
    #===============================#
    
    def train_single(self, input_vector, target_vector):
        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [self.bias]))
        
        #output_vector = []
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T 
        output_hidden = activation(np.dot(self.hidden_weights_in, input_vector))
        if self.bias:
            output_hidden = np.concatenate((output_hidden, [[self.bias]]))
        output_network = activation(np.dot(self.hidden_weights_out, output_hidden))
        
        #Backward phase
        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network * (1.0 - output_network)          
        tmp = self.learning_rate  * np.dot(tmp, output_hidden.T) 
        self.hidden_weights_out += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.hidden_weights_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1,:] 
        else:
            x = np.dot(tmp, input_vector.T)
        self.hidden_weights_in += self.learning_rate * x
        
    def train(self, data_array, labels_one_hot_array, epochs, intermediate_results=False):
        intermediate_weights = []
        for epoch in range(epochs):  
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_one_hot_array[i])
            if intermediate_results:
                intermediate_weights.append((self.hidden_weights_in.copy(), self.hidden_weights_out.copy()))
        
        return intermediate_weights
    
    def run(self, input_vector):
        output_vector = []
        if self.bias:
            # adding bias node
            input_vector = np.concatenate((input_vector, [self.bias]))
            output_vector = np.concatenate((output_vector, [self.bias]))
            
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = activation(np.dot(self.hidden_weights_in, input_vector))
        output_vector = activation(np.dot(self.hidden_weights_out, output_vector))
    
        return output_vector

    #=========================================#
    # Tests the prediction on each element of #
    # the testing data. Prints the precision, #
    # recall, and accuracy.                   #
    #=========================================#
    def test(self, data, labels):
        right, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                right += 1
            else:
                wrongs += 1
        
        return right, wrongs
    
    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm    

    def precision(self, label, confusion_matrix):
        column = confusion_matrix[:, label]
        return confusion_matrix[label, label] / column.sum()
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

#=================================#
# Main method: executed only when #
# the program is run directly and #
# not executed when imported as a #
# module.                         #
#=================================#
def main():
    image_size = 28 
    no_of_different_labels = 10 
    image_pixels = image_size * image_size
    data_path = "./"

    # TODO load training data
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    # TODO load testing data
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")
    
    # removing 0's and 1's from our data
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])
    
    lr = np.arange(10)

    for label in range(10):
        one_hot = (lr==label).astype(np.int)
        print("label: ", label, " in one-hot representation: ", one_hot)
    
    lr = np.arange(no_of_different_labels)

    # transform labels into one hot representation
    train_labels_one_hot = (lr==train_labels).astype(np.float)
    test_labels_one_hot = (lr==test_labels).astype(np.float)

    # we don't want zeroes and ones in the labels neither:
    train_labels_one_hot[train_labels_one_hot==0] = 0.01
    train_labels_one_hot[train_labels_one_hot==1] = 0.99
    test_labels_one_hot[test_labels_one_hot==0] = 0.01
    test_labels_one_hot[test_labels_one_hot==1] = 0.99

    # TODO create a net
    epochs = 10
    
    print("Creating network...")
    network = ANN(no_inputs = image_pixels, no_outputs=10, 
                  no_hidden_layers=100, learning_rate=0.1, bias=None)
    print("Done...")
    
    # TODO call train
    print("Training...")
    weights = network.train(train_imgs, train_labels_one_hot, epochs=epochs, 
                        intermediate_results=True) 
    print("Done...")

    print("Results...")
    for epoch in range(epochs):  
        print("epoch: ", epoch)
        network.hidden_weights_in = weights[epoch][0]
        network.hidden_weights_out = weights[epoch][1]
        
        # TODO call test
        right, wrongs = network.test(train_imgs, train_labels)
        print("accuracy train: ", right / ( right + wrongs))
         
        # TODO call test          
        right, wrongs = network.test(test_imgs, test_labels)
        print("accuracy test: ", right / ( right + wrongs))
     
    print("Confusion matrix for training data...")
    cm_train = network.confusion_matrix(train_imgs, train_labels)
    for i in range(10):
        print("digit:", i, "precision: ", network.precision(i, cm_train), "recall: ", network.recall(i, cm_train))
    print("Done...")
    
    print("Confusion matrix for test data...")
    cm_test = network.confusion_matrix(test_imgs, test_labels)
    for i in range(10):
        print("digit:", i, "precision: ", network.precision(i, cm_test), "recall: ", network.recall(i, cm_test))
    print("Done...")

if __name__ == '__main__':
    main()