################################################################################
# Copyright (C) 2012-2013 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential.                                    #
# Author: Aaron M. Smith                                                       #
################################################################################

import Leap, sys, pickle
sys.path.append("F:\\SecretHandshake\\lib\\libsvm-3.18\\python\\")
from svmutil import *

class FingerJointVector():
    def __init__(self, apiFinger):
        self.joints = [] #four 3-vectors Tip,pip,dip,mcp
        self.compressedRep = [] #unit vector + two angles

        for i in range(4):
            self.joints.append(apiFinger.joint_position(i))

        d01 = apiFinger.joint_position(1) - apiFinger.joint_position(0)
        d12 = apiFinger.joint_position(2) - apiFinger.joint_position(1)
        d23 = apiFinger.joint_position(3) - apiFinger.joint_position(2)
        a02 = d01.angle_to(d12)
        
        # identical to a02 in the current implementation for all but the thumb
        a13 = d12.angle_to(d23)
        
        for i in range(3):
            #self.compressedRep.append(apiFinger.joint_position(0)[i])
            self.compressedRep.append(d01.normalized[i])
        self.compressedRep.append(a02)
        self.compressedRep.append(a13)


class PoseVector():
    def __init__(self, apiHand):
        self.vertices = []
        self.compressedRep = []
        palmCenter = apiHand.palm_position
        palmDir = apiHand.direction

        for finger in apiHand.fingers:
            fingerJointVector = FingerJointVector(finger)
            self.vertices.extend(fingerJointVector.joints)
            self.compressedRep.extend(fingerJointVector.compressedRep)


class PoseListener(Leap.Listener):
    curPose = []
    machine = []
    doRecognition = False

    def on_init(self, controller):
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        if not frame.hands.is_empty:
            # Get the first hand
            hand = frame.hands[0]
            if (hand.is_right == True):
                self.curPose = PoseVector(hand)
                if (self.doRecognition == True):
                    self.recognize(self.curPose.compressedRep)

    def recognize(self, pose):
        self.decisions.append(int(svm_predict([i], [pose], self.machine, "-q")[0][0]))
        print self.decisions[len(self.decisions)-1]
        

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"


def main():
    traind = False

    # Create a training listener and controller
    listener = PoseListener()
    listener.doRecognition = False
    trainingClass = 1
    dataList = []
    classList = []
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until it is killed
    runProgram = True
    while(runProgram):
        print "Type a number to specify the training class,"
        print "Press Enter to store a pose,"
        print "Type 'l' to load data from file,"
        print "Type 't' to train,"
        print "Type 'p' to see the pose classification"
        print "Type 'q' to break." 

        inpt = sys.stdin.readline()
        if (inpt == 'q\n'):
            writeFile = open('Output.txt','w')
            for i in range(len(dataList)):
                dataList[i].append(classList[i])
            pickle.dump(dataList, writeFile)
            writeFile.close()
            # Remove the sample listener when done
            controller.remove_listener(listener)
            runProgram = False
        if (inpt == 't\n'):
            listener.doRecognition = False
            prob = svm_problem(classList, dataList)
            param = svm_parameter()
            #param.kernel_type = LINEAR
            param.kernel_type = RBF
            param.C = 10
            listener.machine = svm_train(prob, param)
            traind = True
        if (inpt == 'p\n'):
            if (traind == False):
                print "The SVM has not been trained."
            else:
                listener.decisions = []
                listener.doRecognition = True
        if (inpt == 'l\n'):
            listener.doRecognition = False
            readFile = open('Output.txt', 'r')
            dataList = pickle.load(readFile)
            classList = []
            for i in range(len(dataList)):
                classList.append(dataList[i].pop())
            readFile.close()
            print "Data loaded.  Length = %d" % len(dataList)    
        if (inpt == '\n'):
            listener.doRecognition = False
            print "Storing hand pose vector with training class %d" % trainingClass
            print "["
            curPose = listener.curPose.compressedRep
            for i in range(25):
                print (curPose[i])
            classList.append(trainingClass)
            dataList.append(curPose)
            print "]"
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

if __name__ == "__main__":
    main()
