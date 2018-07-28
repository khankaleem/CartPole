import GetData
import Train
import Test
import numpy as np

X_train, Y_train = GetData.Populate()
classifier = Train.TrainModel(X_train, Y_train, nb_epoch = 10, batch_size = 10)

Test.PlayGame(classifier)