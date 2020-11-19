
# NamedPipe
import win32file
import sys

import keras
import pandas as pd
import csv
import numpy as np
import os, os.path
from os import listdir



# For visualization
import cv2
import matplotlib.pyplot as plt

# The image size of depth/ir
# Assuming depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED, change it otherwise

# For gray visulization
#MAX_DEPTH_FOR_VIS = 8000.0
#MAX_AB_FOR_VIS = 512.0

clipFrames = 120

runAllAsTest = False

buffer = []
currentdata = []

listOfLabelNames = []

totalModelPerformance = []
listOfLabels =[]

from GestureRecognitionML.Model.Cnn import cnn
from GestureRecognitionML.Model.CnnLstm import cnnlstm
model = cnnlstm(lr=0, bs=0, e=0, loadModel=True, split=1, f='splits120', path="GestureRecognitionML/")


class TestManager:
    def __init__(self, labelName):
        buffer.clear()
        self.pathToTestFiles = "testRecords\\"
        self.path = os.path.abspath(__file__).rsplit("\\", 1)[0] + "\\"
        self.scorePath = self.path + self.pathToTestFiles + "\\TestScore\\" + "TestScores.csv"
        self.data = self.GetData(labelName)

    def AppendPredictionsToTestData(self, data):
        print(len(data))
        print(len(listOfLabels))

        for x in range(0, len(data)):
            listOfLabels[x].append(data[x])

    def GetData(self, labelName):
        df = pd.read_csv(self.path + self.pathToTestFiles + "test" + labelName + ".csv", delimiter=";", decimal=".")
        list = df.values.tolist()
        list.pop(0)
        return list

    def writeHeaderToCSV(self, headerList, values):
        f = open(self.scorePath, 'a')
        with f:
            writer = csv.writer(f, delimiter=';', lineterminator='\n')
            writer.writerow(headerList)
            writer.writerow(values)

    def writeAnalysisToCSV(self, currLabel, currLabelName, testIndex):
        self.writeHeaderToCSV(["", "", "STATS", "FOR", currLabelName], [])
        self.writeHeaderToCSV(["#Predictions", "#Right Predictions", "#Wrong Predictions", "Accuracy", "AVG Precision", "Loss"], [self.numberOfPredictions(currLabel), self.numberOfRightPredictions(currLabel), self.numberOfWrongPredictions(currLabel), self.accuracy(currLabel), self.avgPrecision(currLabel), self.loss(currLabel)])
        totalModelPerformance.append([self.numberOfPredictions(currLabel), self.numberOfRightPredictions(currLabel), self.numberOfWrongPredictions(currLabel), self.accuracy(currLabel), self.avgPrecision(currLabel), self.loss(currLabel)])
        avgPrecisionOtherLabels = self.getPrecisionForOtherLabels(testIndex)
        self.writeHeaderToCSV([], ["", "","AVG", "Precision","For", "Other", "Labels"])
        self.writeHeaderToCSV(avgPrecisionOtherLabels[0], avgPrecisionOtherLabels[1])

    def getPrecisionForOtherLabels(self, testIndex):
        listofNames = listOfLabelNames.copy()
        listofNames.remove(listOfLabelNames[testIndex])
        listOfAvgPrecision = []
        for x in range(0, len(listOfLabels)):
            if testIndex == x:
                continue
            listOfAvgPrecision.append(self.avgPrecision(self.getCurrLabelListFromTestIndex(x)))
        return  [listofNames, listOfAvgPrecision]


    def loss(self, currLabel):
        loss = 0
        for i in currLabel:
            loss += (1-i)
        return loss


    def avgPrecision(self, currLabel):
        avgPrecision = 0

        for i in currLabel:
            avgPrecision += i
        return avgPrecision/len(currLabel)

    def numberOfWrongPredictions(self, currLabel):
        numberOfWrongPredictions = len(currLabel) - self.numberOfRightPredictions(currLabel)
        return numberOfWrongPredictions

    def accuracy(self, currLabel):

        if self.numberOfRightPredictions(currLabel) == 0:
            return 0
        accuracy = self.numberOfRightPredictions(currLabel) / self.numberOfPredictions(currLabel) * 100
        return accuracy

    def numberOfRightPredictions(self, currLabel):
        norp = 0
        for x in range(0, len(currLabel)):
            higestValue = True
            for i in range(0, len(listOfLabels)):
                if currLabel[x] < listOfLabels[i][x]:
                    higestValue = False
            if higestValue:
                norp += 1
        return norp

    def writeModelPerformanceToCSV(self):
        tmp = self.totalModelPerformance()
        self.writeHeaderToCSV(["", "TOTAL", "MODEL", "PERFORMANCE"], [])
        self.writeHeaderToCSV(tmp[0], tmp[1])

    def avgOfList(self, list):
        avg = 0

        for i in list:
            avg += i
        return avg / len(list)

    def totalModelPerformance(self):
        pred =      []
        rigtPred =  []
        wrongpred = []
        acc =       []
        avgPrec =   []
        loss =      []
        for x in totalModelPerformance:
            pred.append(x[0])
            rigtPred.append(x[1])
            wrongpred.append(x[2])
            acc.append(x[3])
            avgPrec.append(x[4])
            loss.append(x[5])

        return[["#Predictions", "#Right Predictions", "#Wrong Predictions", "Accuracy", "AVG Precision", "Loss"], [self.avgOfList(pred), self.avgOfList(rigtPred), self.avgOfList(wrongpred), self.avgOfList(acc), self.avgOfList(avgPrec), self.avgOfList(loss)]]


    def numberOfPredictions(self, currLabel):
        numberOfPredictions = len(currLabel)
        return numberOfPredictions

    def getCurrLabelListFromTestIndex(self, testIndex):
        return listOfLabels[testIndex]

    def WriteDataToCSV(self):
        f = open(self.scorePath, 'a')
        with f:
            w = csv.writer(f, delimiter=';', lineterminator='\n')
            w.writerow([])

        self.writeHeaderToCSV(
            listOfLabelNames, [])
        f = open(self.scorePath, 'a')
        with f:
            writer = csv.writer(f, delimiter=';', lineterminator='\n')
            for x in range(0, len(fastWalk)):
                row = []
                for i in listOfLabels:
                    row.append(listOfLabels[i[x]])

                writer.writerow(row)
            writer.writerow([])
            writer.writerow([])

    def GetNextValue(self):
        if len(self.data) == 0:
            return None
        return self.data.pop(0)

    def clearAllList(self):
        for x in listOfLabels:
            x.clear()



class DataHandler:
    def __init__(self):
        self.testManager = None


    def WriteTotalModelPerformance(self):
        self.testManager.writeModelPerformanceToCSV()

    def handleData(self, filhandle, testIndex):
        self.testManager = TestManager(listOfLabelNames[testIndex])
        OldStamp = -1.
        loss = 0
        acc = 0
        labelIndex = 0


        while True:
            if not runAllAsTest:
                request_msg = "Request bodyInfo"
                win32file.WriteFile(fileHandle, request_msg.encode())
                inputData = win32file.ReadFile(fileHandle, 1168)  # (32*6+1) * 4bytes
                data = np.frombuffer(inputData[1], dtype="float32", count=-1, offset=0)
            else:
                print(testFileName)
                data = self.testManager.GetNextValue()
                if data == None:
                    break

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
                    #print(rs.shape)
                    encode = listOfLabelNames
                    predict = model.predict(shape, clipFrames, True)[0]

                    if runAllAsTest:
                        self.testManager.AppendPredictionsToTestData(predict)

                    print('PREDICTION:')
                    encodedLabels = []
                    for i in range(0, len(predict)):
                        encodedLabels.append((encode[i], predict[i]))

                    def sortByCertainty(label):
                        return -label[1]

                    encodedLabels.sort(key=sortByCertainty)
                    for encode in encodedLabels:
                        print(encode)
                    print("*********************************'")
                    loss += (1 - predict[labelIndex])
                    if encodedLabels[0] == labelIndex:
                        acc += 1
                    print(loss)
                #else:
                    #print(len(buffer))

            key = cv2.waitKey(1)
            if key == 27:  # Esc key to stop
                break

        print('out of while')
        if not runAllAsTest:
            win32file.CloseHandle(fileHandle)
        else:
            self.testManager.writeHeaderToCSV(["Model Type", "#Labels", "Label", "Label test video length", "#Batch size", "Epoch", "Frames pr clip", "Date"],
                                         ["CNNLSTM", "31",  listOfLabelNames[testIndex], "20 min", "200", "400", clipFrames, "31-10-2020"])
            self.testManager.writeHeaderToCSV([],[])
            self.testManager.writeAnalysisToCSV(self.testManager.getCurrLabelListFromTestIndex(testIndex), listOfLabelNames[testIndex], testIndex)
            self.testManager.writeHeaderToCSV([], [])
            self.testManager.writeHeaderToCSV([], [])
            #testManager.WriteDataToCSV()'''
            self.testManager.clearAllList()


if __name__ == "__main__":

    # Create pipe client

    DataHandler = DataHandler()
    fileHandle = None

    if not runAllAsTest:
        fileHandle = win32file.CreateFile("\\\\.\\pipe\\mynamedpipe",
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0, None,
            win32file.OPEN_EXISTING,
            0, None)

    filenames = listdir((os.path.abspath(__file__).rsplit("\\", 1)[0]) + "\\testRecords")
    listOfTestNames = [filename for filename in filenames if filename.endswith(".csv")]
    for x in listOfTestNames:
        listOfLabelNames.append(((x.split(".")[0]).split("test")[1]))
        listOfLabels.append([])

    print(listOfLabelNames)
    print(len(listOfLabels))



    testIndex = 0
    for testFileName in listOfTestNames:
        DataHandler.handleData(fileHandle, testIndex)
        testIndex += 1
    DataHandler.WriteTotalModelPerformance()




