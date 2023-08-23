# 2023/5/30 Acc = 92.01% Basic NN model

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from torchsummary import summary

torch.manual_seed(7)

def check_all_attr():
    df = pd.read_csv('dataset/newdropdata.csv')
    data = df.values
    attr = df.columns[1:]
    n = attr.tolist()
    offset = 5
    print("")
    print("==Data attributes included==")
    for i in range(0, len(n), offset):	# 인덱스 0을 시작으로 n 길이의 전 까지, 10씩
	    print(f'{i+1: 3d} ~{i+offset: 3d} | '," , ".join(["'"+x+"'" for x in n[i:i+offset]]))
    
    
    
def dataload_preprocessing(drops=[]):
    df = pd.read_csv('dataset/newdropdata.csv')
    data = df.values
    attr = df.columns[1:]
    # NaN data pre-processing
    for i, item in enumerate (data):
         for j, item2 in enumerate(item):
             if math.isnan(item2) : data[i,j] = 0.0

    y = np.asarray(data[:, 0])
    y2 = np.abs(1 - y)
    label = np.vstack([y, y2]).transpose().tolist()

    
    # The attributes which all zero values have are removed
    drop_attr = ['pp1hnum18',
     'pc1nfcons',
     'pc1npcons',
     'pc1nfccons',
     'pc1npccons',
     'pa1toth',
     'pa1flunch',
     'pa1datt',
     'pa1trans',
     'pa1capa',
     'pa1ell',
     'pa1see',
     'pa1alt',
     'pa1dropp',
     'pa1app',
     'pa1hispp',
     'pa1blackp',
     'pa1asianp',
     'pa1americp',
     'pa1rep9p',
     'pa1ret9p',
     'pa1senc',
     'pa1sena',
     'pa1workp',
     'pa1armyp',
     'pa1teachfn',
     'pa1yprins',
     'pa1teachm',
     'pa1teachs',
     'pa1hdisci',
     'pa1ptabs']

    # df.drop(columns =drop_attr , inplace=True)
    
    normalize_attr1 = ['b1hnum',
     'b1gpa9',
     'b1credit',
     'b1skipclass',
     'b1inschsusp',
     'ps1ncredit',
     'ps1agpa',
     'ps1gpa',
     'pp1hnum',
     'pp1osib',
     'pa1yprin',
     'pa1hteacher',
     'pa1hmang',
     'pa1hemang',
     'pa1hmont',
     'pa1hteach',
     'pa1hpart',
     'pa1hmeets',
     'pa1hpaper',
     'pa1hother',
    ]
    normalize_attr2 = ['W3W1W2STU','W1STUDENT']
    normalize_attr3 = ['ps1clcost']
    normalize_attr4 = ['pc1load']
    normalize_attr5 = ['b1mtest', 'ps1mtest']
    normalize_attr6 = ['b1schoolcli', 'X3CLASSES', 'X3WORK', 'S3FAMILY', 'S3APPFAFSA']


    t_df = df[normalize_attr1]
    scaled_X = MinMaxScaler().fit_transform(t_df.values)
    df[normalize_attr1] = scaled_X

    t_df = df[normalize_attr2]
    scaled_X = MinMaxScaler().fit_transform(t_df.values)
    df[normalize_attr2] = scaled_X

    t_df = df[normalize_attr3]
    scaled_X = MinMaxScaler().fit_transform(t_df.values)
    df[normalize_attr3] = scaled_X

    t_df = df[normalize_attr4]
    scaled_X = MinMaxScaler().fit_transform(t_df.values)
    df[normalize_attr4] = scaled_X

    t_df = df[normalize_attr5]
    scaled_X = StandardScaler().fit_transform(t_df.values)
    df[normalize_attr5] = scaled_X

    t_df = df[normalize_attr6]
    scaled_X = StandardScaler().fit_transform(t_df.values)
    df[normalize_attr6] = scaled_X
    
    X_df = df.drop(columns=['everdrop'])
    y_df = pd.DataFrame(label)
    print("")
    print("Number of data attributes: ",len(X_df.shape[1]))
    if len(drops)>0:
        print(len(drops)," data attributes are removed.")
        X_df.drop(columns=drops,inplace=True)
    print("Number of data attributes in Traing/Testing process: ",X_df.shape[1])
    
    return X_df, y_df

def dataload_preprocessing_svm(drops=[]):
    df = pd.read_csv('dataset/newdropdata.csv')
    data = df.values
    attr = df.columns[1:]
    # NaN data pre-processing
    for i, item in enumerate(data):
        for j, item2 in enumerate(item):
            if math.isnan(item2): data[i, j] = 0.0

    y = np.asarray(data[:, 0])
    # y2 = np.abs(1 - y)
    # label = np.vstack([y, y2]).transpose().tolist()

   # The attributes which all zero values have are removed
    drop_attr = ['pp1hnum18',
     'pc1nfcons',
     'pc1npcons',
     'pc1nfccons',
     'pc1npccons',
     'pa1toth',
     'pa1flunch',
     'pa1datt',
     'pa1trans',
     'pa1capa',
     'pa1ell',
     'pa1see',
     'pa1alt',
     'pa1dropp',
     'pa1app',
     'pa1hispp',
     'pa1blackp',
     'pa1asianp',
     'pa1americp',
     'pa1rep9p',
     'pa1ret9p',
     'pa1senc',
     'pa1sena',
     'pa1workp',
     'pa1armyp',
     'pa1teachfn',
     'pa1yprins',
     'pa1teachm',
     'pa1teachs',
     'pa1hdisci',
     'pa1ptabs']

    # df.drop(columns =drop_attr , inplace=True)
    
    normalize_attr1 = ['b1hnum',
     'b1gpa9',
     'b1credit',
     'b1skipclass',
     'b1inschsusp',
     'ps1ncredit',
     'ps1agpa',
     'ps1gpa',
     'pp1hnum',
     'pp1osib',
     'pa1yprin',
     'pa1hteacher',
     'pa1hmang',
     'pa1hemang',
     'pa1hmont',
     'pa1hteach',
     'pa1hpart',
     'pa1hmeets',
     'pa1hpaper',
     'pa1hother',
    ]
    normalize_attr2 = ['W3W1W2STU','W1STUDENT']
    normalize_attr3 = ['ps1clcost']
    normalize_attr4 = ['pc1load']
    normalize_attr5 = ['b1mtest', 'ps1mtest']
    normalize_attr6 = ['b1schoolcli', 'X3CLASSES', 'X3WORK', 'S3FAMILY', 'S3APPFAFSA']


    t_df = df[normalize_attr1]
    scaled_X = MinMaxScaler().fit_transform(t_df.values)
    df[normalize_attr1] = scaled_X

    t_df = df[normalize_attr2]
    scaled_X = MinMaxScaler().fit_transform(t_df.values)
    df[normalize_attr2] = scaled_X

    t_df = df[normalize_attr3]
    scaled_X = MinMaxScaler().fit_transform(t_df.values)
    df[normalize_attr3] = scaled_X

    t_df = df[normalize_attr4]
    scaled_X = MinMaxScaler().fit_transform(t_df.values)
    df[normalize_attr4] = scaled_X

    t_df = df[normalize_attr5]
    scaled_X = StandardScaler().fit_transform(t_df.values)
    df[normalize_attr5] = scaled_X

    t_df = df[normalize_attr6]
    scaled_X = StandardScaler().fit_transform(t_df.values)
    df[normalize_attr6] = scaled_X
    
    X_df = df.drop(columns=['everdrop'])
    y_df = pd.DataFrame(y)
    print("")
    print("Number of data attributes: ",len(X_df.shape[1]))
    if len(drops)>0:
        print(len(drops)," data attributes are removed.")
        X_df.drop(columns=drops,inplace=True)
    print("Number of data attributes in Traing/Testing process: ",X_df.shape[1])
    return X_df, y_df

def show_data(X, y):

    y = np.argmax(y, axis=1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    clf = pca.fit_transform(X)

    fig1 = plt.figure(figsize=(8,8))
    ax = fig1.add_subplot(1,1,1, projection='3d')

    ax.scatter(clf[y==0,0], clf[y ==0,1], clf[y ==0,2], c='g', marker='o', label='y=0')
    ax.scatter(clf[y==1,0], clf[y ==1,1], clf[y ==1,2], c='r', marker='o', label='y=1')
    plt.title("Data distribution")
    plt.legend()
    plt.savefig('data_distribution.png')


def plot_decision_regions_2class(X, y):
    y = np.argmax(y.values, axis=1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    clf = pca.fit_transform(X)

    h = .02
    x_min, x_max = clf[:, 0].min() - 0.1, clf[:, 0].max() + 0.1
    y_min, y_max = clf[:, 1].min() - 0.1, clf[:, 1].max() + 0.1
    # y_min, y_max = y.min() - 0.1 , y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

    fig2 = plt.figure(figsize=(8, 8))
    plt.plot(clf[y == 0, 0], clf[y == 0, 1], 'o', label='y=0')
    plt.plot(clf[y == 1, 0], clf[y == 1, 1], 'ro', label='y=1')
    plt.title("decision region")
    plt.legend()
    plt.savefig('decision_region.png')

# def show_model(model, data):
#     #### Summarize and Visualize NN model in graph
#     from torchviz import make_dot
#     make_dot(model(data[0][0]), params=dict(model.named_parameters())).render('model_short', format='png')
#     summary(model, (len(data), input_dim))

class Net(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out, dropout = 0.2):
        super(Net, self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(D_in, H)
        self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(H, 64)
        self.linear3 = nn.Linear(64, D_out)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x)).to(device=self.device)
        x = self.dp1(x).to(device=self.device)
        x = torch.tanh(self.linear2(x)).to(device=self.device)
        x = self.linear3(x).to(device=self.device)

        return x

# Create dataset object
class Edu_Data(Dataset):
    # Constructor
    def __init__(self, X,y):
        self.x = X
        self.y = y
        self.len = X.shape[0]
    # Getter
    def __getitem__(self, index):
        x = self.x.loc[index]
        y = self.y.loc[index]
        return torch.Tensor(x), torch.Tensor(y) #torch.tensor(y, dtype=torch.long)
    # Get Length
    def __len__(self):
        return self.len

# Define the train model
def train(model, criterion, train_loader, optimizer):
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)
    
    train_loss = 0
    success = 0
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        y = y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        pred = model(x)
        loss = criterion(pred, y)
        train_loss += loss
        result = pred.argmax(dim=1, keepdim=True)
        label = y.argmax(dim=1, keepdim=True)
        success += result.eq(label).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    return train_loss/len(train_loader.dataset), success/len(train_loader.dataset)

def validate(model, val_loader):
    model.eval()
    loss = 0
    success =0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            y = y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
            pred = model(x)
            loss += criterion(pred, y)
            result = pred.argmax(dim=1, keepdim=True)
            label = y.argmax(dim=1, keepdim=True)
            success += result.eq(label).sum().item()

    return loss/len(val_loader.dataset) , success/len(val_loader.dataset)
 

from sklearn.metrics import f1_score  
def test(model, test_loader):
    model.eval()
    loss = 0
    success =0
    total_result = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            y = y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            pred = model(x)
            loss += criterion(pred, y)
                
            result = pred.argmax(dim=1, keepdim=True)
            label = y.argmax(dim=1, keepdim=True)
            success += result.eq(label).sum().item()
                                  
            
            for r, l in zip(result, label):
                if r == l:
                    total_result.append('pass')
                else:
                    total_result.append('fail')

    return loss/len(test_loader.dataset), success/len(test_loader.dataset), total_result

import torchmetrics
from torchmetrics.classification import BinaryRecall

def test_various_metric(model, test_loader):
    model.eval()
    loss = 0
    success =0
    total_f1 = 0
    total_recall = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            y = y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            pred = model(x)
            
            result = pred.argmax(dim=1, keepdim=True).squeeze()
            label = y.argmax(dim=1, keepdim=True).squeeze()
            
            f1 = torchmetrics.F1Score(task="binary")
            recall = BinaryRecall()

            total_f1+=f1(result, label)
            total_recall+=recall(result,label)
            # print(recall(result,label))
            # print(f1_)
    return total_f1/len(test_loader), total_recall/len(test_loader)



learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()

best_validation_loss = float('inf')

# def precision(outputs, labels):
#     op = outputs.cpu()
#     la = labels.cpu()
#     _, preds = torch.max(op, dim=1)
#     return torch.tensor(precision_score(la,preds, average=‘weighted’))