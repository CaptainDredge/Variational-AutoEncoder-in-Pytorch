from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import models
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
from tensorboardX import SummaryWriter
from glob import glob
from util import *
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

from vae import VAE, ShallowVAE


parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    is_cuda = True
else:
    is_cuda = False

BATCH_SIZE = args.batch_size
EPOCH = args.epochs
LOG_INTERVAL=args.log_interval
path = '../PetImages/'
kwargs = {'num_workers': 3, 'pin_memory': True} if is_cuda else {}

simple_transform = transforms.Compose([transforms.Resize((224,224))
                                       ,transforms.ToTensor(), transforms.Normalize([0.48829153, 0.45526633, 0.41688013],[0.25974154, 0.25308523, 0.25552085])])
train = ImageFolder(path+'train/',simple_transform)
valid = ImageFolder(path+'valid/',simple_transform)
train_data_gen = torch.utils.data.DataLoader(train,shuffle=True,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])
valid_data_gen = torch.utils.data.DataLoader(valid,batch_size=BATCH_SIZE,num_workers=kwargs['num_workers'])

dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
dataloaders = {'train':train_data_gen,'valid':valid_data_gen}

model = ShallowVAE(latent_variable_size=500, nc=3, ngf=224, ndf=224, is_cuda=is_cuda)

#model = VAE(BasicBlock, [2, 2, 2, 2], latent_variable_size=500, nc=3, ngf=224, ndf=224, is_cuda=is_cuda)

if is_cuda:
    model.cuda()
    
reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):

    MSE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(epoch):

    model.train()
    train_loss = 0
    batch_idx = 1
    for data in dataloaders['train']:
        # get the inputs
        inputs, _ = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(inputs)
        #print(inputs.data.size())
        inputs.data = unnormalize(inputs.data,[0.48829153, 0.45526633, 0.41688013],[0.25974154, 0.25308523, 0.25552085])

        #print("input max/min"+str(inputs.max())+"  "+str(inputs.min()))
        #print("recon input max/min"+str(recon_batch.max())+"  "+str(recon_batch.min()))
        loss = loss_function(recon_batch, inputs, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), (len(dataloaders['train'])*128),
                100. * batch_idx / len(dataloaders['train']),
                loss.data[0] / len(inputs)))
        batch_idx+=1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(dataloaders['train'])*BATCH_SIZE)))
    return train_loss / (len(dataloaders['train'])*BATCH_SIZE)

def test(epoch):
    model.eval()
    test_loss = 0
    for data in dataloaders['valid']:
        # get the inputs
        inputs, _ = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        recon_batch, mu, logvar = model(inputs)
        inputs.data = unnormalize(inputs.data,[0.48829153, 0.45526633, 0.41688013],[0.25974154, 0.25308523, 0.25552085])
        test_loss += loss_function(recon_batch, inputs, mu, logvar).data[0]
        if((epoch+1)%10==0):
            torchvision.utils.save_image(inputs.data, './imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
            torchvision.utils.save_image(recon_batch.data, './imgs/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)

    test_loss /= (len(dataloaders['valid'])*128)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

writer = SummaryWriter('runs/exp-1')
since = time.time()
for epoch in range(EPOCH):
    train_loss = train(epoch)
    test_loss = test(epoch)
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('test_loss',test_loss, epoch)
    torch.save(model.state_dict(), './models/Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))
time_elapsed = time.time() - since    
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))