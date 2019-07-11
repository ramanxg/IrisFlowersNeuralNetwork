from neuralnetwork import NeuralNetwork
from matrix import Matrix
from matplotlib import pyplot as plt
import pandas as pd
import random
import numpy as np

#processes and separates data into a train and test set
def create_sets(data):
    #create a list of the rows separating inputs and outputs
    data_list = []
    for _, rows in data.iterrows():
        #create a target output for each neural network
        target = [0] * 3
        target[int(rows.species)] = 1
        data_list.append([[ rows.sepal_length, rows.sepal_width, rows.petal_length, rows.petal_width], target ])
    #randomly divide the data between a train set and a test set
    random.shuffle(data_list)
    test_size = 0.2
    train_data = data_list[:-int(test_size*len(data_list))]
    test_data = data_list[-int(test_size*len(data_list)):]
    return train_data, test_data

def train_nn(num_input, num_hid, num_out, data_list, num_iterations=1000):
    #Neural Network with 2 inputs, 6 hidden, and 3 outputs
    nn = NeuralNetwork(num_input, num_hid, num_out)
    for i in range(num_iterations):
        for data in data_list:
            #print(data[0], data[1])
            nn.train(data[0], data[1])
        random.shuffle(data_list)
        print("Training Data #", i)
    return nn

def test_nn(nn, test_data):
    color_dict = {0: "red", 1: "blue", 2: "green"}
    for flower in test_data:
        #calculate the mean squared error for each prediction
        guess = nn.feedforward(flower[0])
        error = Matrix.vector(flower[1]) - guess
        error.elementmul(error)
        mse = sum(Matrix.transpose(error).toList())
        if mse > 0.05:
            print(mse)
        ind = np.argmax(Matrix.transpose(guess).toList())
        plt.scatter(flower[0][2], flower[0][3], c = color_dict[ind])

    plt.show()
   


def main():
    df = pd.read_csv("iris.data.txt")
    df.loc[df['class']=='Iris-virginica','species']=0
    df.loc[df['class']=='Iris-versicolor','species']=1
    df.loc[df['class']=='Iris-setosa','species'] = 2

    train_data, test_data = create_sets(df)

    nn = train_nn(4, 6, 3, train_data, 1000)

    test_nn(nn, test_data) 


main()

