from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import os
import numpy as np
import scipy 
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4,4), stride=1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 16, 3, 1)
        self.dropout1 = nn.Dropout2d(.5)
        self.dropout2 = nn.Dropout2d(.25)
        self.dropout3 = nn.Dropout2d(.125)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.relu(x)
        #x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    #cm = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            '''
            print((target-pred.flatten()).numpy())
            wrong = np.argwhere((target-pred.flatten()).numpy() != 0).tolist()
            print(wrong)

            if len(wrong) >= 9:
                fig, axarr = plt.subplots(3, 3)
                for i in range(9):
                    img = data[wrong[i][0], 0]
                    axarr[int(i/3), i%3].imshow(img.detach().numpy())
                
                    axarr[int(i/3), i%3].set_title("target:"+str(target[wrong[i]].numpy()[0])+" pred:"+str(pred.flatten()[wrong[i]].numpy()[0]))
                plt.tight_layout(pad=2.0)
                plt.show()
            '''
            #cm += confusion_matrix(target.numpy(), pred.flatten().numpy())
            test_num += len(data)
    #print(cm)
    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))
    return test_loss, correct, test_num

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        # Visualize conv filter
        kernels = model.conv1.weight.detach()
        print(kernels.size())
        fig, axarr = plt.subplots(3,3)
        for idx in range(kernels.size(0)):
            axarr[int(idx/3),idx%3].imshow(kernels[idx].squeeze())
        plt.show()

        outputs=[]
        def hook(module, input, output):
            outputs.append(output.detach().numpy())
        model.fc1.register_forward_hook(hook)
        targets=[]
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            targets.append(target.numpy())
        print(np.shape(outputs))
        o = []
        t = []
        for i in range(np.shape(outputs)[0]):
            for j in range(np.shape(outputs)[1]):
                o.append(outputs[i][j])
                t.append(targets[i][j])
        print(np.shape(o))
        print(np.shape(t))
        o_e = TSNE(n_components=2).fit_transform(o)
        tsne = plt.scatter(o_e[:, 0], o_e[:, 1], c=t, cmap=plt.cm.tab10, s=.5, alpha=.5)
        plt.legend(*tsne.legend_elements(),loc="lower left", title="Numbers")
        plt.show()

        distances = scipy.spatial.distance.cdist(o[0:4], o, metric='euclidean')
        imgs_idx=[]
        for i in range(4):
            imgs_idx.append(distances[i].argsort()[:8])

        fig, axarr = plt.subplots(4,8)
        for i in range(4):
            j = 0
            for idx in imgs_idx[i]:
                axarr[i,j].imshow(test_dataset[idx][0].squeeze())
                j+=1
        plt.show()

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    #transforms.Resize((30,30)),
                    #transforms.RandomCrop((28,28)),
                    #transforms.RandomRotation(degrees=5),
                    #transforms.ColorJitter(brightness=.5),
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    indices = []
    subset_indices_train = []
    subset_indices_valid = []
    np.random.seed(430)
    for i in range(10):
        indices.append([j for j in range(len(train_dataset)) if train_dataset[j][1] == i])
        for index in indices[i]:
            rand = np.random.rand()
            if rand > .15:
                subset_indices_train.append(index) 
            else:
                subset_indices_valid.append(index)

    subset_indices_train_sub = []
    for i in subset_indices_train:
        rand = np.random.rand()
        if rand >= 0.0:
            subset_indices_train_sub.append(i)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train_sub)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    #model = ConvNet().to(device)
    model = Net().to(device)

    # Try different optimzers here [Adadelta, Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    epoch_train_val = {}
    epoch_train_val['epoch']=[]
    epoch_train_val['train_loss']=[]
    epoch_train_val['val_loss']=[]
    for epoch in range(1, args.epochs + 1):
        epoch_train_val['epoch'].append(epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        print("Accuracy on training set:")
        train_loss, train_c, train_tn = test(model, device, train_loader)
        epoch_train_val['train_loss'].append(train_loss)
        print("Accuracy on validation set:")
        val_loss, val_c, val_tn = test(model, device, val_loader)
        epoch_train_val['val_loss'].append(val_loss)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    ptrain,=plt.plot(epoch_train_val['epoch'], epoch_train_val['train_loss'], c='g')
    pval,=plt.plot(epoch_train_val['epoch'], epoch_train_val['val_loss'], c='r')
    plt.xlabel("epoch #")
    plt.ylabel("Loss")
    plt.legend([ptrain,pval],['train','validation'])
    plt.show()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
