# -*- coding: utf-8 -*-
u'''BVH Parser Module
read(parse), write and other operate functions'''
import csv
import numpy as np

# BVH file read & parse to Node & Motion Matrix
def readBVH(bvhAbsPath):
    u'''Read BVH file and parse to each parts
    return tuple of (RootNode, MotionSeries, Frames, FrameTime)
    '''
    with open(bvhAbsPath) as bvhFile:
        # "HIERARCHY"part
        hierarchyStack = []
        nodeIndex = 0
        frameIndex = 0
        tmpNode = None

        for line in bvhFile:
            # cutting line terminator
            line = line.rstrip("\n")
            line = line.rstrip("\r")

            # parse BVH Hierarcy
            if "{" in line:
                hierarchyStack.append(tmpNode)
                tmpNode = newNode
                continue
            if "}" in line:
                tmpNode = hierarchyStack.pop()
                continue
            if ("JOINT" in line) or ("ROOT" in line):
                newNode = BVHNode(line.rsplit(None, 1)[1], nodeIndex, frameIndex)
                nodeIndex += 1
                if tmpNode != None:
                    tmpNode.addChild(newNode)
                else:
                    tmpNode = newNode
                continue
            if "OFFSET" in line:
                if tmpNode.fHaveSite == True:
                    tmpNode.site.extend([float(data) for data in line.split(None, 3)[1:]])
                else:
                    tmpNode.offset.extend([float(data) for data in line.split(None, 3)[1:]])
                continue
            if "CHANNELS" in line:
                tmpNode.chLabel.extend(line.split(None, 7)[2:])
                frameIndex += len(tmpNode.chLabel)
                continue
            if "End Site" in line:
                tmpNode.fHaveSite = True
                continue
            if "MOTION" in line:
                break
        else:
            raise ValueError("This File is not BVH Motion File.")

        isNewLine = lambda string: not string.rstrip("\n").rstrip("\r")
        # get frames "Frames: xxx"
        line = bvhFile.readline()
        while isNewLine(line):
            line = bvhFile.readline()
        frmNum = int(line.rsplit(None, 1)[1])

        # get frameTime "Frame Time: x.xx"
        line = bvhFile.readline()
        while isNewLine(line):
            line = bvhFile.readline()
        frmTime = float(line.rsplit(None, 1)[1])

        # get "MOTION"part (List of List)
        motionSeries = [([float(data) for data in line.split()] if not isNewLine(line) else None) for line in bvhFile]
        try:
            while True:
                motionSeries.remove(None)
        except ValueError:
            pass

    return tmpNode, motionSeries, frmNum, frmTime

def writeBVH(path, root, motionSeries, frameNum, frameTime):
    '''Write BVH format
    path : file directory + file name
    root : root of skeleton joints
    motionSeries : motion part, rows is frame, columns is rotation angles
    '''
    def _writeNodeInfo(hierPartStr, tmpNode, hierNum):
        u'''nested function: writeBVH/_writeNodeInfo, buffering hierarchy strings'''
        indent = "  " * (hierNum + 1)
        strOffsetLine = indent + "OFFSET "
        strOffsetLine += " ".join([str(num) for num in tmpNode.offset]) + "\n"
        hierPartStr.append(strOffsetLine)

        strChannelsLine = indent + "CHANNELS " + str(len(tmpNode.chLabel))
        strChannelsLine += " " + " ".join(tmpNode.chLabel) + "\n"
        hierPartStr.append(strChannelsLine)

        if tmpNode.fHaveSite:
            hierPartStr.append(indent + "End Site" + "\n")
            hierPartStr.append(indent + "{" + "\n")
            strOffsetLine = indent + "  OFFSET "
            strOffsetLine += " ".join([str(num) for num in tmpNode.site]) + "\n"
            hierPartStr.append(strOffsetLine)
            hierPartStr.append(indent + "}" + "\n")
        else:
            for child in tmpNode.childNode:
                hierPartStr.append(indent + "JOINT " + child.nodeName + "\n")
                hierPartStr.append(indent + "{" + "\n")
                _writeNodeInfo(hierPartStr, child, hierNum+1)
                hierPartStr.append(indent + "}" + "\n")

    # writeBVH main code
    with open(path, "w") as dstFile:
        # HIERARCHY writing
        hierPartStr = []
        hierPartStr.append("HIERARCHY\n")
        hierPartStr.append("ROOT " + root.nodeName + "\n")
        hierPartStr.append("{\n")
        _writeNodeInfo(hierPartStr, root, 0)
        hierPartStr.append("}\n")
        dstFile.writelines(hierPartStr)

        # MOTION writing
        dstFile.write("MOTION\n")
        dstFile.write("Frames: " + str(frameNum) + "\n")
        dstFile.write("Frame Time: " + str(frameTime) + "\n")
        writer = csv.writer(dstFile, lineterminator="\n", delimiter=' ')
        writer.writerows(motionSeries)

def splitMotionPart(src, fRootHaveBoth=True):
    u'''Separate the motion part into position and rotation channels
    fRootHaveBoth : root has position/rotation channels in both sides
    return : tuple of (Position Series, Rotation Series)
    '''
    srcMat = np.array(src)
    if fRootHaveBoth:
        dstPosMat = srcMat[:, 0:6]
        dstRotMat = srcMat[:, 0:6]
    else:
        dstPosMat = srcMat[:, 0:3]
        dstRotMat = srcMat[:, 3:6]

    for i in range(6, len(src[0]), 6):
        dstPosMat = np.c_[dstPosMat, srcMat[:, i  :i+3]]
        dstRotMat = np.c_[dstRotMat, srcMat[:, i+3:i+6]]

    return dstPosMat.tolist(), dstRotMat.tolist()

def combineMotionPart(srcPosition, srcRotation, fRootHaveBoth=True):
    u'''Combine the motion part position and rotation channels
    fRootHaveBoth : if root has position/rotation channels in both sides, set true
    return : Combined motion series
    '''
    srcPosMat = np.array(srcPosition)
    srcRotMat = np.array(srcRotation)

    if len(srcPosition) != len(srcRotation):
        raise ValueError("srcPosition and srcRotation must be same size.")

    if fRootHaveBoth:
        dstMat = srcPosition[:, 0:6]
        for i in range(6, len(srcPosition), 3):
            dstMat = np.c_[dstMat, srcPosMat[:, i:i+3]]
            dstMat = np.c_[dstMat, srcRotMat[:, i:i+3]]
    else:
        for i in range(0, len(srcPosition), 3):
            dstMat = np.c_[dstMat, srcPosMat[:, i:i+3]]
            dstMat = np.c_[dstMat, srcRotMat[:, i:i+3]]

    return dstMat.tolist()

def chChannelComposition(root, mode, fRootHaveBoth=True):
    '''change the channel list of hierarchy part
    fRootHaveBoth : if ignore root joint, set true
    '''
    if not mode in ["POSITION", "ROTATION", "ALL"]:
        raise ValueError("Available mode are \"POSITION\", \"ROTATION\" or\"ALL\"")

    if mode is "ALL":
        # position / rotation
        for node in root.getNodeList():
            if len(node.chLabel) == 6:
                continue
            tmpChLabel = node.chLabel
            node.chLabel = ["Xposition", "Yposition", "Zposition"]
            node.chLabel.extend(tmpChLabel)
    else:
        # position only or rotation only
        # Root Node
        if not fRootHaveBoth:
            if mode is "POSITION":
                root.chLabel = root.chLabel[0:3]
            else:
                root.chLabel = root.chLabel[3:6]

        # Other Nodes
        for node in root.getNodeList():
            if node.nodeIndex == 0: # ROOT Node
                continue
            if mode is "POSITION":
                if node.chLabel[0] is "Xposition":
                    node.chLabel = node.chLabel[0:3]
                else:
                    # chLabel rotation only
                    node.chLabel = ["Xposition", "Yposition", "Zposition"]
            else:
                # mode is "ROTATION"
                if len(node.chLabel) == 6:
                    node.chLabel = node.chLabel[3:6]


class BVHNode:
    u'''BVH Skeleton Joint Structure'''
    def __init__(self, nodeName, nodeIndex, frameIndex):
        self.nodeName = nodeName
        self.nodeIndex = nodeIndex
        self.frameIndex = frameIndex
        self.offset = []
        self.chLabel = []
        self.childNode = []
        self.fHaveSite = False
        self.site = []

    def addChild(self, childNode):
        u'''add child joint'''
        self.childNode.append(childNode)

    def getChannelIndex(self, channelName):
        u'''return non-negative index 0, 1, ..., or None(Error) '''
        try:
            return self.chLabel.index(channelName)
        except ValueError:
            return None

    def getNodeI(self, index):
        u'''return BVHNode instanse or None(Error)'''
        if index == self.nodeIndex:
            return self
        if self.fHaveSite:
            return None
        for node in self.childNode:
            retNode = node.getNode(index)
            if retNode != None:
                return retNode

    def getNodeN(self, name):
        u'''return BVHNode instanse or None(Error)'''
        node = None
        if name == self.nodeName:
            return self
        else:
            for child in self.childNode:
                node = child.getNodeN(name)
                if node is not None:
                    return node
        return None

    def getNodeList(self):
        u'''return list of BVHNode (element order : order of appearance in the BVH file)'''
        nodelist = [self]
        if self.fHaveSite:
            return nodelist
        for child in self.childNode:
            nodelist.extend(child.getNodeList())
        return nodelist
