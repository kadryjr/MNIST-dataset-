import PIL
from keras.datasets import mnist
from matplotlib import pyplot
import numpy
from sklearn.neighbors import KNeighborsClassifier


TrainingSamples = 10000
TestingSamples = 1000

def new_data(data):
    height = 6
    width = 4
    new_data = []
    for image in range(len(data)):
        image = PIL.Image.fromarray(numpy.uint8(data[image]))
        feature_vector = numpy.array(crop(image, height, width))
        new_data.append(feature_vector)
    new_data = numpy.vstack(new_data)
    return new_data



def crop(image, height, width):
    feature_vector = []
    imgwidth, imgheight = image.size
    for i in range(imgwidth // width):
        for j in range(imgheight // height):
            box = (j * height, i * width, (j + 1) * height, (i + 1) * width)
            block = image.crop(box)
            block = numpy.asarray(block)
            centre_x, centre_y = centre(block, width, height)
            feature_vector.append(centre_x)
            feature_vector.append(centre_y)
    return feature_vector



def centre(image, image_width, image_height):
    centre_x = 0
    centre_y = 0
    pixels = 0
    for i in range(image_width):
        for j in range(image_height):
            centre_x = centre_x + i * image[i][j]
            centre_y = centre_y + j * image[i][j]
            pixels = pixels + image[i][j]
    centre_x = centre_x / pixels if pixels > 0 else 0
    centre_y = centre_y / pixels if pixels > 0 else 0
    return centre_x, centre_y






(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('train_X: ' + str(train_X.shape))
print('train_Y: ' + str(train_y.shape))
print('test_X:  ' + str(test_X.shape))
print('test_Y:  ' + str(test_y.shape))
pyplot.subplot(330 + 1)
pyplot.imshow(train_X[10], cmap=pyplot.get_cmap('gray'))
pyplot.show()
train_X = new_data(train_X[0:TrainingSamples])
test_X = new_data(test_X[0:TestingSamples])
train_Y = train_y[0:TrainingSamples]
test_Y = test_y[0:TestingSamples]
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(train_X, train_Y)
score = knn.score(test_X, test_Y)
print("Accuracy = ", score * 100.0, "%")