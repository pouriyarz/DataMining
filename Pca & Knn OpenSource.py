#load data

import numpy as np
import matplotlib.pyplot as plt
import math



from os import listdir
from os.path import isfile, join
import cv2
path = 'att_faces'
dirs = [directory for directory in listdir(path)]

x = []
y = []

for v in dirs:
    face_path = join(path, v)
    images = [img for img in listdir(face_path) if img.endswith(".pgm")]
    label = int(v)
    for img in images:
        y.append(label)
        image =img_array = cv2.imread(join(face_path,img), cv2.IMREAD_GRAYSCALE)
        resized_image  = cv2.resize(img_array, (32, 32))
        img_flatten = resized_image.reshape(-1)
        x.append(img_flatten)

X = np.array(x)  # data
y = np.array(y)  # target



#split

def split(data, target, random = False):
    """
    data -> X
    target -> Y
    """
    
    train_data = list()
    train_label = list()
    test_data = list()
    test_label = list()
    
    ratio = .5
    images, features = data.shape
    persons = len(dirs)  # count of persons
    images_for_each_person = images // persons
    
    for person in range(persons):
        # slicing:
        start_train = person * images_for_each_person  # e.g. first -> 0
        stop_train = person * images_for_each_person + int(ratio * images_for_each_person)  # e.g. first -> 5
        start_test = stop_train  # e.g. first -> 5
        stop_test = person * images_for_each_person + images_for_each_person  # e.g. first -> 10
        
        if random == True:
            np.random.shuffle(data[start_train:stop_test])
        
        # split data and label for train and test:
        train_data.extend(data[start_train:stop_train])
        train_label.extend(target[start_train:stop_train])
        test_data.extend(data[start_test:stop_test])
        test_label.extend(target[start_test:stop_test])
        # !!! test_label is same as train_label !!!

    # convert to 'ndarray':
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    

    return train_data, train_label, test_data, test_label

# pca

def pca(data, count):
    n_samples, n_features = data.shape

    stds = np.array([np.std(data,axis=0)])
    means = np.array([np.mean(data,axis=0)])

    X_standard = np.zeros((n_samples,n_features))


    for row in range(n_samples):
        for col in range(n_features):
            X_standard[row,col] = (data[row,col] - means[:,col])/stds[:,col] # standardize
    

    cov = np.cov(X_standard.transpose())
    eigen_value, eigen_vector = np.linalg.eig(cov)
    eigen_value = np.real(eigen_value)
    eigen_vector = np.real(eigen_vector)

    # Make a list of (eigenvalue, eigenvector) tuples:
    eigen_list = list()
    for index in range(len(eigen_value)):
        eigen_list.append((np.abs(eigen_value[index]), eigen_vector[:, index]))
        
    eigen_list.sort(key=lambda k: k[0], reverse=True)

    # stack eigen vectors horizontally with eigen vectors with high eigen value are in first columns:
    vecs = list()
    for val, vec in eigen_list:
        vecs.append(vec)

    eigen_vec = np.stack(vecs, axis=1)

    return eigen_vec [:,:count],means,stds

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getKNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

import operator
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))     


def run_KNN(X_train_new, y_train, X_test_new, y_test):
    y_train = y_train.reshape(200,1)
    TrainSet1 = np.append(X_train_new, y_train, axis=1)
    y_test = y_test.reshape(200,1)
    TestSet1 = np.append(X_test_new, y_test, axis=1)
    T = []
    TE = []
    for i in range(200):
        T.append(TrainSet1[i, :])
    for i in range(200):
        TE.append(TestSet1[i, :])
    k = 1
    prediction = []
    for i in range(200):
        neighbors = getKNeighbors(T, TE[i], k)
        pred = getResponse(neighbors)
        prediction.append(pred)
    accuracy = getAccuracy(TE, prediction)
    return accuracy

def main():
    X_train, y_train, X_test, y_test = split(X, y, random=False)
    X_new, means, stds = pca(X_train, 5)
    X_train_new = np.dot(np.divide(np.subtract(X_train, means), stds), X_new)  # standardize
    X_test_new = np.dot(np.divide(np.subtract(X_test, means), stds), X_new)
    print('shape of X_train:')
    print(X_train.shape,'\n')
    
    X_new , means , stds = pca(X_train,5)
    print('shape of X_pca:',)
    print(X_new.shape,'\n')

    X_train_new = np.dot(np.divide(np.subtract(X_train,means),stds),X_new)   # standardize
    
    print('shape of X_train_pca:')
    print(X_train_new.shape,'\n')
    
    print('data reduced!','\n')# standardize
    np.save('X_train_pca',X_train_new)
    print('X_train_pca:')
    print(X_train_new,'\n')
    
    n_feature = X_train.shape[1]
    n = 5

    mean_accuracy = []
    indices = []
    X_train, y_train, X_test, y_test = split(X, y, random=False)
    X_new, means, stds = pca(X_train, -1)
    print('model with straight split:','(please wait..!)','\n')
    while n <= n_feature:   
        X_train_new = np.dot(np.divide(np.subtract(X_train, means), stds), X_new[:,:n])  # standardize
        X_test_new = np.dot(np.divide(np.subtract(X_test, means), stds), X_new[:,:n])  # standardize
        # fit a knn classifier
        accuracy = run_KNN(X_train_new, y_train, X_test_new, y_test)
        print('number_of_feature:', n)
        print('accuracy:', accuracy, '\n')
        mean_accuracy.append(accuracy)
        indices.append(n)
        n += 5

    indices = np.array(indices)
    print('maximum of accuracy:', np.max(mean_accuracy))
          
    # draw a chart
    plt.figure(1)
    plt.plot(indices, mean_accuracy)
    plt.title(" Modeling With Straight Data")
    plt.ylabel('accuracy')
    plt.xlabel('number of feature')
    plt.grid
    plt.gray()
    plt.show()      
          
    n = 5
    n_indices = []
    acc_avg=[]
    accuracy_of_nth_feature = []
    print('model with random split:','(please wait..!)','\n')
    while n <= n_feature:
        for j in range (20):
            X_train, y_train, X_test, y_test = split(X, y, random=True)
            X_new, means, stds = pca(X_train, n)
            X_train_new = np.dot(np.divide(np.subtract(X_train, means), stds), X_new[:,:n])  # standardize
            X_test_new = np.dot(np.divide(np.subtract(X_test, means), stds), X_new[:,:n])  # standardize
            # fit a knn classifier
            accuracy = run_KNN(X_train_new, y_train, X_test_new, y_test)
            accuracy_of_nth_feature.append(accuracy)
            
        n_indices.append(n)
        accuracy_avg = np.mean(accuracy_of_nth_feature)
        acc_avg.append(accuracy_avg)
        print('number_of_feature:', n)
        print('average of accuracy:', accuracy_avg, '\n')
        n += 5
    n_indices = np.array(n_indices)
    accuracy_of_nth_feature.index(np.max(accuracy_of_nth_feature))
    print('maximum of accuracy:', np.max(accuracy_of_nth_feature), '\n',
          'number of selected features in max_accuracy:',
          n_indices[accuracy_of_nth_feature.index(np.max(accuracy_of_nth_feature))])

    # draw a chart
    plt.figure(2)
    plt.plot(n_indices, accuracy_of_nth_feature)
    plt.title(" Modeling With Random Data")
    plt.ylabel('Average accuracy')
    plt.xlabel('number of features')
    plt.grid
    plt.gray()
    plt.show()
if __name__ == "__main__":
    main()



