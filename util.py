import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import Layout2ImageE2E
import sys

def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataLabelDataset(Dataset):
    def __init__(self, data, label):
        self.matrix = data
        self.label = label

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        features = self.matrix[idx]
        label = self.label[idx]
        return features, label
    

def loadDataset(which):
    

    if which == 'iccad2012':
    
        case2 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/2/train.gds", '2', False)
        label2 = Layout2ImageE2E.extractLabel()
        case3 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/3/train.gds", '3', False)
        label3 = Layout2ImageE2E.extractLabel()
        case4 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/4/train.gds", '4', False)
        label4 = Layout2ImageE2E.extractLabel()
        case5 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/5/train.gds", '5', False)
        label5 = Layout2ImageE2E.extractLabel()

        traindata = case2 + case3 + case4 + case5
        trainlabel = label2 + label3 + label4 + label5

        case1 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/1/test.gds", '1', False)
        label1 = Layout2ImageE2E.extractLabel()
        case2 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/2/test.gds", '2', False)
        label2 = Layout2ImageE2E.extractLabel()
        case3 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/3/test.gds", '3', False)
        label3 = Layout2ImageE2E.extractLabel()
        case4 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/4/test.gds", '4', False)
        label4 = Layout2ImageE2E.extractLabel()
        case5 = Layout2ImageE2E.extractFeature(8, "./benchmark/gds/5/test.gds", '5', False)
        label5 = Layout2ImageE2E.extractLabel()

        testdata = case1 + case2 + case3 + case4 + case5
        testlabel = label1 + label2 + label3 + label4 + label5
    elif which == 'iccad2019-1':
        traindata = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/train/training_dataset.gds", 't', False)
        trainlabel = Layout2ImageE2E.extractLabel()

        testdata = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test1/testing_dataset_1.gds", 't', False)
        testlabel = Layout2ImageE2E.extractLabel()
    elif which == 'iccad2019-2':

        traindata = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/train/training_dataset.gds", 't', False)
        trainlabel = Layout2ImageE2E.extractLabel()

        case1 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip1_test.gds", 't', False)
        label1 = Layout2ImageE2E.extractLabel()
        case2 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip2_test.gds", 't', False)
        label2 = Layout2ImageE2E.extractLabel()
        case3 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip3_test.gds", 't', False)
        label3 = Layout2ImageE2E.extractLabel()
        case4 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip4_test.gds", 't', False)
        label4 = Layout2ImageE2E.extractLabel()
        case5 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip5_test.gds", 't', False)
        label5 = Layout2ImageE2E.extractLabel()
        case6 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip6_test.gds", 't', False)
        label6 = Layout2ImageE2E.extractLabel()
        case7 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip7_test.gds", 't', False)
        label7 = Layout2ImageE2E.extractLabel()
        case8 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip8_test.gds", 't', False)
        label8 = Layout2ImageE2E.extractLabel()
        case9 = Layout2ImageE2E.extractFeature(8, "./benchmark/iccad2019/test2/clip9_test.gds", 't', False)
        label9 = Layout2ImageE2E.extractLabel()

        testdata = case1 + case2 + case3 + case4 + case5 + case6 + case7 + case8 + case9
        testlabel = label1 + label2 + label3 + label4 + label5 + label6 + label7 + label8 + label9
    else:
        print('Incorrect dataset name.')
        sys.exit()



    aug_traindata = []
    aug_trainlabel = []

    for d, l in zip(traindata, trainlabel):
        if l == 1:
            d1  = torch.rot90(d, k=1, dims=(0, 1))
            d2  = torch.rot90(d, k=1, dims=(0, 1))
            d3 = torch.flip(d, dims=[1])
            d4 = torch.flip(d, dims=[0])
            aug_traindata.extend([d, d1, d2, d3, d4])
            aug_trainlabel.extend([l, l, l, l, l])
        else:
            aug_traindata.append(d)
            aug_trainlabel.append(l)
    trainset = DataLabelDataset(aug_traindata, aug_trainlabel)
    testset = DataLabelDataset(testdata, testlabel)

    return trainset, testset, aug_trainlabel