import torch
from model import ANNModel
import os
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import time
from torchvision.transforms import GaussianBlur
from torch.utils.data import Dataset, DataLoader
import util
import sys




def process(which, pretrain):
    s = 55
    sigma = 3.3
    num_epochs = 15
    batch_size = 32
    lr = 0.001
    util.set_seed(0)

    trainset, testset, trainlabel = util.loadDataset(which)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)


    model = ANNModel().to(device)
    

    hotspot_weight = len(np.where(np.array(trainlabel) == 1)[0])
    nonhotspot_weight = len(np.where(np.array(trainlabel) == 0)[0])
    total = hotspot_weight + nonhotspot_weight
    hotspot_weight = 1 - hotspot_weight / total
    nonhotspot_weight = 1 - nonhotspot_weight / total

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([nonhotspot_weight, hotspot_weight])).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) 
    gaussian_blur = GaussianBlur(kernel_size=(s, s), sigma=sigma)

    if pretrain == '0':

        for epoch in range(num_epochs):
            model.train()  
            running_loss = 0.0
            
            for inputs, labels in tqdm(train_loader):

                inputs = inputs.to(device)
                blurred = gaussian_blur(inputs).to(dtype=torch.float32).reshape(-1, 1, 480, 480)
                labels = labels.to(device=device).to(dtype=torch.uint8).reshape(-1)

                optimizer.zero_grad()

                outputs = model(blurred)
                
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                optimizer.step()

                running_loss += loss.item()
            scheduler.step()
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainset)}')

    elif pretrain == '1':
        model = torch.load('model_'+which+'.pth')

    model.eval()  
    pred = []
    ls = []
    start = time.time()
    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            blurred = gaussian_blur(inputs).to(dtype=torch.float32).reshape(-1, 1, 480, 480)
            labels = labels.to(device=device, dtype=torch.uint8).reshape(-1)
            outputs = model(blurred)
            predicted = torch.argmax(outputs.data, 1)
            pred += predicted.cpu().tolist()
            ls += labels.cpu().tolist()
    end = time.time()   
    print(f'Inference time: {end - start}s')
    acc = accuracy_score(ls, pred)
    recall = recall_score(ls, pred)
    cm = confusion_matrix(ls, pred)
    fa = cm[0, 1]
    print(f'Accuracy on test set: {acc}')
    print(f'Recall on test set: {recall}')
    print(f'False Alarm on test set: {fa}')
    torch.save(model, 'model_'+which+'.pth')


if __name__ == "__main__":
    which = sys.argv[1] 
    pretrain = sys.argv[2]
    process(which, pretrain)