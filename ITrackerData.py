import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import pandas as pd


MEAN_PATH = './'

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor.sub(self.meanImg)


class ITrackerData(data.Dataset):
    def __init__(self, dataPath, split = 'train', imSize=(224,224), gridSize=(25, 25)):

        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize

        print('Loading iTracker dataset...')


        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']

        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])




        if split == 'test':
            #mask = self.metadata['labelTest']
            self.dataset_path = "./dataset/test/"
        # elif split == 'val':
        #     self.mask = self.metadata['labelVal']
        else:
            #mask = self.metadata['labelTrain']
            self.dataset_path = "./dataset/train/"

        #self.indices = np.argwhere(mask)[:,0]
        #print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))
        self.indices = len(os.listdir(self.dataset_path+"appleFace"))
        print('Loaded iTracker dataset split "%s" with %d records...' % (split, self.indices))

        self.coor = []
        with open(self.dataset_path+'images_label.txt','r') as f:
            for line in f.readlines():
                x1, y1 = line.split(" ")[:2]
                self.coor.append([x1,y1])


        gridcsv = pd.read_csv(self.dataset_path+"faceGrid.csv", encoding = 'utf_8_sig')
        self.gridlist = gridcsv.values.tolist()

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        #index = self.indices[index]

        imFacePath = os.path.join(self.dataset_path, 'appleFace/%d.png' % index)
        imEyeLPath = os.path.join(self.dataset_path, 'appleLeftEye/%d.png' % index)
        imEyeRPath = os.path.join(self.dataset_path, 'appleRightEye/%d.png' % index)

        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        #gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)
        gaze = np.array([self.coor[index][0],self.coor[index][1]], np.float32)

        # print('gaze:' + str(gaze))
        #print('gaze shape:' + str(gaze.shape))

        #faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])
        faceGrid = np.array(self.gridlist[index][1:], np.float32)

        #print('faceGrid['+str(index)+']'+str(faceGrid))
        #print('faceGridShape'+str(faceGrid.shape))

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)
        # print('tensor gaze:' + str(gaze))
        # print('tensor gaze shape:' + str(gaze.shape))


        return row, imFace, imEyeL, imEyeR, faceGrid, gaze


    def __len__(self):
        return self.indices
