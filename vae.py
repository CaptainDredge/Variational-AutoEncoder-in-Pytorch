import torch
from torch.autograd import Variable
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class VAE(nn.Module):
    def __init__(self, block, layers, latent_variable_size, nc, ngf, ndf, is_cuda=False):
        super(VAE, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.is_cuda = is_cuda
        #Encoder
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 512)
        self.fc1 = nn.Linear(512 , latent_variable_size)
        self.fc2 = nn.Linear(512 , latent_variable_size)
        
        #Decoder
        self.fc3 = nn.Linear(latent_variable_size, 500)
        self.fc4 = nn.Linear(500, 14*14*32)
        self.deconv1 = nn.ConvTranspose2d(32,64, kernel_size=3, stride =2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64,32, kernel_size=3, stride =2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32,16, kernel_size=3, stride =2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16,3, kernel_size=3, stride =2, padding=1, output_padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        #print((x>0.000).sum())
        w_mean = self.fc1(x)
        w_std  = self.fc2(x)
        return w_mean, w_std

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        x = self.fc3(z)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        #print(x.size())
        #print((x>0.000).sum())
        x = x.view(-1,32,14,14)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.sigmoid(x)

        return x

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar
    
    
    
class ShallowVAE(nn.Module):
    def __init__(self,latent_variable_size, nc, ngf, ndf, is_cuda=False):
        super(ShallowVAE, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.is_cuda = is_cuda
        #Encoder
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,32, kernel_size=3,stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(14*14*32, 512)
        self.fc1 = nn.Linear(512 , latent_variable_size)
        self.fc2 = nn.Linear(512 , latent_variable_size)
        
        #Decoder
        self.fc3 = nn.Linear(latent_variable_size, 500)
        self.fc4 = nn.Linear(500, 14*14*32)
        self.deconv1 = nn.ConvTranspose2d(32,64, kernel_size=3, stride =2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64,32, kernel_size=3, stride =2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32,16, kernel_size=3, stride =2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16,3, kernel_size=3, stride =2, padding=1, output_padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)
        #print((x>0.000).sum())
        w_mean = self.fc1(x)
        w_std  = self.fc2(x)
        return w_mean, w_std

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        x = self.fc3(z)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        #print(x.size())
        #print((x>0.000).sum())
        x = x.view(-1,32,14,14)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.sigmoid(x)

        return x

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar