import numpy as np
from guessNumberNetwork import ngn

trainingDataFile = open("DataSets/mnist_train.csv", 'r')
scorecard = []
right = 0.0
# test are written on each lines, so they're read one at a time
for data in trainingDataFile.readlines():
    data_values = data.split(',')
    # input must be resized
    input_data = (np.asfarray(data_values[1:]) / 255.0) * 0.99 + 0.01  # l'input deve essere una lista
    # image_data = input_data.reshape((28, 28))
    # set the expected output
    target_result = int(data_values[0])
    target_data = np.zeros(ngn.onodes) + 0.01
    target_data[target_result] = 0.99
    # print of the image with matplotlib
    # print("<Image:")
    # plt.imshow(image_data, cmap = "Greys")
    output_result = ngn.query(input_data)
    risposta = np.argmax(output_result) 
    ngn.train(input_data, target_data)
    if risposta == target_result:
        scorecard.append(1)
        right = right + 1
    else:
        scorecard.append(0)
    pass

# print(scorecard)
print(right / len(scorecard) * 100, "% corretti.", sep="")
trainingDataFile.close()

ngn.exportWeights("weights.txt")
# computed weights are stored so they can be used later
