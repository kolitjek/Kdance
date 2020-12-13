import numpy as np
import os, os.path
from os import listdir
import cv2
import pandas as pd

clipFrames = 30
runAllAsTest = True
testMIDI = True
buffer = []
currentdata = []
listOfLabelNames = []
totalModelPerformance = []
listOfLabels =[]

print('loading player from osc_interface...')
from GestureRecognitionML.midi.osc_interface import Player
print('loading model...')
from GestureRecognitionML.Model.CnnLstm import cnnlstm

print('init model')
model = cnnlstm(lr=0, bs=0, e=0, loadModel=True, split=1, f='splits120', path="GestureRecognitionML/")


class TestManager:

    def __init__(self, labelName):
        buffer.clear()
        self.pathToTestFiles = "testRecords/"
        self.path = os.path.abspath(__file__).rsplit("/", 1)[0] + "/"
        self.scorePath = self.path + self.pathToTestFiles + "/TestScore/" + "TestScores.csv"
        self.data = self.GetData(labelName)

    def GetData(self, labelName):
        df = pd.read_csv(self.path + self.pathToTestFiles + "test" + labelName + ".csv", delimiter=";", decimal=".")
        list = df.values.tolist()
        list.pop(0)
        return list

    def GetNextValue(self):
        if len(self.data) == 0:
            return None
        return self.data.pop(0)




class DataHandler:
    def __init__(self):
        self.liveInterface = Player()
        self.testManager = None
        self.liveInterface.play()
        self.play

    def handleData(self, filhandle, testIndex):
        self.testManager = TestManager(listOfLabelNames[testIndex])
        OldStamp = -1.

        while True:
            data = self.testManager.GetNextValue()

            if (OldStamp != data[288]):
                OldStamp = data[288]
                if (len(buffer) == clipFrames):
                    buffer.pop(0)
                elif not runAllAsTest:
                    buffer.append(data[:-3])  # removes the last 3 elements from the list (x, y ,z)
                else:
                    buffer.append(data)

                rs = np.asarray(buffer)

                if (len(buffer) == clipFrames):
                    shape = np.reshape(rs, (rs.shape[0], rs.shape[1]))
                    encode = listOfLabelNames
                    predict = model.predict(shape, clipFrames, True)[0]

                    encodedLabels = []
                    for i in range(0, len(predict)):
                        encodedLabels.append((encode[i], predict[i]))

                    def sortByCertainty(label):
                        return -label[1]

                    encodedLabels.sort(key=sortByCertainty)

                    self.liveInterface.updatePredictions(encodedLabels)
                    self.liveInterface.clock()

            key = cv2.waitKey(1)
            if key == 27:  # Esc key to stop
                break


if __name__ == "__main__":
    print('loading data handler')
    DataHandler = DataHandler()

    filenames = listdir((os.path.abspath(__file__).rsplit("/", 1)[0]) + "/testRecords")
    filenames.sort()
    listOfTestNames = [filename for filename in filenames if filename.endswith(".csv")]

    for x in listOfTestNames:
        listOfLabelNames.append(((x.split(".")[0]).split("test")[1]))
        listOfLabels.append([])

    testIndex = 0
    for testFileName in listOfTestNames:
        DataHandler.handleData(None, testIndex)
        testIndex += 1

