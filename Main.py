################################################################################
# Copyright (C) 2014                                                           #
#                                                                              #
# Author: Aaron M. Smith                                                       #
################################################################################

import sys, pickle, math, serial
sys.path.append("C:\\LocalLibraries\\lib\\x86\\")
sys.path.append("C:\\LocalLibraries\\lib\\libsvm-3.18\\python\\")
import Leap
from svmutil import *
from numpy import random

# This is a workaround for the Leap.Vector.angle_to() function which occasionally returns nan.
def angleTo(v1,v2):
    denom = v1.magnitude_squared * v2.magnitude_squared
    prec = 1e-6
    if (denom <= prec):
        return 0.0
    else:
        val = (v1.dot(v2) / denom)
        if (math.fabs(val) > (1.0 - prec)):
            return 0.0
        return math.acos(val)

class FingerJointVector():
    # This contructor packages a finger into a list of four 3-vectors as well as a compressed,
    # hand-relative representation consisting of four angles, all normalized into the range [0,1]
    def __init__(self, apiFinger, palmCenter, palmNormal):
        self.joints = [] #four 3-vectors Tip,pip,dip,mcp
        self.compressedRep = [] #four angles

        # Append joints into the joint list
        for i in range(4):
            self.joints.append(apiFinger.joint_position(i))

        # Important joint->joint vectors
        dc0 = apiFinger.joint_position(0) - palmCenter
        d01 = apiFinger.joint_position(1) - apiFinger.joint_position(0)
        d12 = apiFinger.joint_position(2) - apiFinger.joint_position(1)
        d23 = apiFinger.joint_position(3) - apiFinger.joint_position(2)

        dc0 = dc0.normalized
        cross1 = palmNormal.cross(dc0)
        # Basis for the plane perp. to dc0
        cross1 = cross1.normalized
        cross2 = dc0.cross(cross1)

        # Projection of d01 onto the plane perp. to dc0
        proj = d01 - (dc0 * (d01.dot(dc0)))
        proj = proj.normalized

        # Spherical coordinate angles for d01 rel. to dc0:
        # polar angle:
        ac1 = angleTo(dc0, d01) / (math.pi)
        # azimuthal angle
        acn = angleTo(cross1, proj)
        acn = math.copysign(acn, proj.dot(cross2))
        acn = (acn + (math.pi)) / (2 * math.pi)
        # Middle rel. finger angle
        a02 = angleTo(d01, d12) * 2 / (math.pi)
        # Final rel. finger angle
        a13 = angleTo(d12, d23) * 2 / (math.pi)

        self.compressedRep.append(ac1)
        self.compressedRep.append(acn)
        self.compressedRep.append(a02)
        self.compressedRep.append(a13)


class PoseVector():
    # This constructor appends the five compressed representations into a single pose representation,
    # as well as storing all 20 joint coordinates into a single vertex list
    # The compressed pose representation is 20-dimensional and is invariant under the action of SE(3) on the hand
    def __init__(self, apiHand):
        self.vertices = []
        self.compressedRep = []
        palmCenter = apiHand.palm_position
        palmDir = apiHand.direction
        palmNormal = apiHand.palm_normal

        for finger in apiHand.fingers:
            fingerJointVector = FingerJointVector(finger, palmCenter, palmNormal)
            self.vertices.extend(fingerJointVector.joints)
            self.compressedRep.extend(fingerJointVector.compressedRep)


class PoseListener(Leap.Listener):
    curPose = []
    machine = []
    doRecognition = False

    def on_init(self, controller):
        print "Initialized Leap Controller"

    def on_connect(self, controller):
        print "Connected Leap Controller"

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected Leap Controller"

    def on_exit(self, controller):
        print "Exited Leap Control"

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        if not frame.hands.is_empty:
            # Get the first hand
            hand = frame.hands[0]
            if (hand.is_right == True):
                # Store the current pose vector
                self.curPose = PoseVector(hand)
                if (self.doRecognition == True):
                    # Do pose prediction from the svm
                    self.recognize(self.curPose.compressedRep)
                else:
                    self.filter_reccognitions(0)
            else:
                self.filter_reccognitions(0)
        else:
            self.filter_reccognitions(0)

    # Predict the current pose
    def recognize(self, pose):
        self.decisions.append(int(svm_predict([i], [pose], self.machine, "-q")[0][0]))
        #print self.decisions[len(self.decisions)-1]
        self.filter_reccognitions(self.decisions[len(self.decisions)-1])

    # Apply filtering to recognitions
    def filter_reccognitions(self, pose):
        self.recentFrequency.setdefault(pose, 0)
        self.recentFrequency[pose] += 1
        if pose != self.currentGuess and self.recentFrequency[pose] >= self.threshold + max([self.recentFrequency[k] if k != pose and k != 0  and k!= self.currentGuess else 0 for k in self.recentFrequency]):
            self.currentGuess = pose
            self.recentFrequency = {}
            if self.currentGuess != 0:
                self.seenPoses.append(self.currentGuess)
                if ((len(self.seenPoses) - len(self.password)) >= 0):
                    print self.seenPoses[len(self.seenPoses) - len(self.password):]
                else:
                    print self.seenPoses
                if self.password == self.seenPoses[len(self.seenPoses) - len(self.password):]:
                    print "Open sesame"
                    self.ser.write("O")

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"

# Divides the data into numDivs pieces in order to run the cross validation scheme.
# The selection into bins is done at random.
def DivideDataRandom(dataByClass, numDivs, numClasses, dividedData, dividedClasses):
    sampleSize = map(lambda i : len(dataByClass[i]) // numDivs, range(numClasses))
    print sampleSize
    for j in range(numClasses):
        random.permutation(dataByClass[j])

    for i in range(numDivs):
        for j in range(numClasses):
            for k in range(sampleSize[j]):
                dividedData[i].append(dataByClass[j][sampleSize[j] * i + k])
                dividedClasses[i].append(j)

# Computes the cross valication percentage associated to the pair of params C,gamma
def CrossValidate(numDivs, dataByClass, numClasses, C, gamma):
    dividedData = []
    dividedClasses = []
    for i in range(numDivs):
        dividedData.append([])
        dividedClasses.append([])
    DivideDataRandom(dataByClass, numDivs, numClasses, dividedData, dividedClasses)

    param = svm_parameter()
    param.kernel_type = RBF
    param.C = C
    param.gamma = gamma

    crossValCount = 0
    totSamples = 0
    for i in range(numDivs):
        curClassList = []
        curDataList = []
        for j in range(numDivs):
            if (j != i):
                curClassList.extend(dividedClasses[j])
                curDataList.extend(dividedData[j])

        prob = svm_problem(curClassList, curDataList)
        machine = svm_train(prob, param)

        for j in range(len(dividedData[i])):
            totSamples = totSamples + 1
            if (int(svm_predict([i], [dividedData[i][j]], machine, "-q")[0][0]) == int(dividedClasses[i][j])):
                crossValCount = crossValCount + 1

    if (totSamples > 0):
        return (float(crossValCount) / float(totSamples))
    else:
        return 0.0

# Performs a grid search for the best (C, gamma) parameters determined by the best
# cross validation percentage
def GridSearchParams(nC, nG, dataByClass, numClasses):
    numDivs = 10
    bestRatio = 0.0
    bestC = 0
    bestG = 0
    # Logarithmic grid search
    #for C in [math.pow(2, x - (nC//2)) for x in range(nC // 2)]:
    #    for G in [math.pow(2, y - nG + 10) for y in range(nG)]:
    #        ratio = CrossValidate(numDivs, dataByClass, numClasses, C, G)
    #        if (ratio > bestRatio):
    #            bestRatio = ratio
    #            bestC = C
    #            bestG = G
    # Linear grid search
    for C in [.750 + .05/float(nC) * x for x in range(nC)]:
        for G in [228 + 38/float(nG) * y for y in range(nG)]:
            ratio = CrossValidate(numDivs, dataByClass, numClasses, C, G)
            if (ratio > bestRatio):
                bestRatio = ratio
                bestC = C
                bestG = G
    print 'Best ratio: {}'.format(bestRatio)
    print 'Best gamma: {}'.format(bestG)
    print 'Best C: {}'.format(bestC)
    return [bestRatio, bestC, bestG]

def main():
    # Number of training classes
    numClasses = 9
    # Did we train the SVM yet?
    traind = False
    # Create a training listener and controller
    listener = PoseListener()
    # Prediction mode
    listener.doRecognition = False
    # Default prediction
    listener.currentGuess = 0
    # Filtering params
    listener.threshold = 20
    listener.recentFrequency = {}
    # Poses recognized
    listener.seenPoses = []
    # Password
    listener.password = [1,2,6,3,5,1]
    # Serial port communication
    listener.ser = serial.Serial(2) #COM3
    # Default training label
    trainingClass = 1
    # Default margin coefficient
    CVal = .770
    # Default gamma value
    GVal = 252.2
    # Main data list
    dataList = []
    # Associated class labels
    classList = []
    # Data stoed by classList as index
    dataByClass = []
    for i in range(numClasses + 1):
        dataByClass.append([])
    # The Leap controller
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Main IO loop:
    # Keep this process running until it is killed
    runProgram = True
    while(runProgram):
        print "Type a number to specify the training class (currently = %d)," % trainingClass
        print "Press Enter to add a pose to the current data list,"
        print "Type 'l' to load data from file,"
        print "Type 'v' to compute best training parameters via cross validation"
        print "Type 't' to train with current parameters,"
        print "Type 'p' to see the pose classification,"
        print "Type 's' to store the current data list to file,"
        print "Type 'q' to quit."

        inpt = sys.stdin.readline()
        # Quit:
        if (inpt == 'q\n'):
            # Remove the sample listener when done
            controller.remove_listener(listener)
            runProgram = False
        # Save
        if (inpt == 's\n'):
            writeFile = open('Output.txt','w')
            for i in range(len(dataList)):
                dataList[i].append(classList[i])
            pickle.dump(dataList, writeFile)
            writeFile.close()
        # Cross Validate
        if (inpt == 'v\n'):
            listener.doRecognition = False
            foundParams = GridSearchParams(10, 10, dataByClass, numClasses)
            CVal = foundParams[1]
            GVal = foundParams[2]
        # Train
        if (inpt == 't\n'):
            listener.doRecognition = False
            print "Training with C= {}, and gamma = {}".format(CVal, GVal)
            prob = svm_problem(classList, dataList)
            param = svm_parameter()
            param.kernel_type = RBF
            param.C = CVal
            param.gamma = GVal
            listener.machine = svm_train(prob, param)
            traind = True
        # Predict
        if (inpt == 'p\n'):
            if (traind == False):
                print "The SVM has not been trained."
            else:
                listener.decisions = []
                listener.doRecognition = True
        # Load
        if (inpt == 'l\n'):
            listener.doRecognition = False
            readFile = open('Output.txt', 'r')
            dataList = pickle.load(readFile)
            classList = []
            for i in range(len(dataList)):
                curClass = dataList[i].pop()
                classList.append(curClass)
                dataByClass[curClass].append(dataList[i])
            readFile.close()
            print "Data loaded.  Length = %d" % len(dataList)

        # Enter data
        if (inpt == '\n'):
            if (listener.doRecognition == False):
                print "Storing hand pose vector with training class %d" % trainingClass
                print "["
                curPose = listener.curPose.compressedRep
                for i in range(20):
                    print (curPose[i])
                classList.append(trainingClass)
                dataList.append(curPose)
                dataByClass[trainingClass].append(curPose)

                print "]"
            else:
                listener.doRecognition = False
        # Change trainging label
        if (inpt == '0\n'):
            trainingClass = 0
            print "Traing Class set to %d" % trainingClass
        if (inpt == '1\n'):
            trainingClass = 1
            print "Traing Class set to %d" % trainingClass
        if (inpt == '2\n'):
            trainingClass = 2
            print "Traing Class set to %d" % trainingClass
        if (inpt == '3\n'):
            trainingClass = 3
            print "Traing Class set to %d" % trainingClass
        if (inpt == '4\n'):
            trainingClass = 4
            print "Traing Class set to %d" % trainingClass
        if (inpt == '5\n'):
            trainingClass = 5
            print "Traing Class set to %d" % trainingClass
        if (inpt == '6\n'):
            trainingClass = 6
            print "Traing Class set to %d" % trainingClass
        if (inpt == '7\n'):
            trainingClass = 7
            print "Traing Class set to %d" % trainingClass
        if (inpt == '8\n'):
            trainingClass = 8
            print "Traing Class set to %d" % trainingClass
        if (inpt == '9\n'):
            trainingClass = 9
            print "Traing Class set to %d" % trainingClass
        if (inpt == 'c\n'):
            print "Closing box"
            listener.ser.write('C')

if __name__ == "__main__":
    main()
