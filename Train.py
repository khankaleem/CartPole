import keras
from keras.models import Sequential
from keras.layers import Dense

def TrainModel(X_train, Y_train, nb_epoch = 10, batch_size = 10):
    
    #build neural network
    classifier = Sequential()
    
    classifier.add(Dense(output_dim = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
    
    classifier.add(Dense(output_dim = 8, kernel_initializer = 'uniform', activation = 'relu'))
    
    classifier.add(Dense(output_dim = 2, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    #compile neural network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #Fit neural network
    classifier.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch)   

    #return model
    return classifier