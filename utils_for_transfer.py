import torch
from torch import nn
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import time
import random
from skimage import transform


class VAE(nn.Module):
    def __init__(self, KERNEL=3,PADDING=1):
        super(VAE, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convt1=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.convt2=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.convt3=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.convt4=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)

        self.conv_seq1 = nn.Sequential( nn.Conv2d(1,64,kernel_size=KERNEL,padding=PADDING),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64,64,kernel_size=KERNEL,padding=PADDING),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.conv_seq2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(128),
                                       nn.ReLU(inplace=True))
        self.conv_seq3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(256),
                                       nn.ReLU(inplace=True))
        self.conv_seq4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.conv_seq5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(1024),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(1024, 1024, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(1024),
                                       nn.ReLU(inplace=True))


        self.deconv_seq1 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(512, 512, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(512),
                                       nn.ReLU(inplace=True))
        self.deconv_seq2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(256),
                                       nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                         nn.InstanceNorm2d(256),
                                         nn.ReLU(inplace=True),
                                        )

        self.down4fc1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(256),
                                      nn.Tanh())
        self.down4fc2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(256),
                                      nn.Tanh())
        self.segdown4_seq = nn.Sequential(nn.Conv2d(256, 3, kernel_size=KERNEL, padding=PADDING),)

        self.deconv_seq3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(128),
                                       nn.ReLU(inplace=True))

        self.down2fc1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(128),
                                      nn.Tanh())
        self.down2fc2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=KERNEL, padding=PADDING),
                                      nn.InstanceNorm2d(128),
                                      nn.Tanh())
        self.segdown2_seq = nn.Sequential(nn.Conv2d(128, 3, kernel_size=KERNEL, padding=PADDING),)

        self.deconv_seq4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout2d(p=0.5),
                                       nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True),)

        self.fc1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                 nn.InstanceNorm2d(64),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                 nn.InstanceNorm2d(64),
                                 nn.Tanh())

        self.deconv_seq5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 3, kernel_size=KERNEL, padding=PADDING))
        self.soft = nn.Softmax2d()

        self.upsample2 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.segfusion = nn.Sequential(nn.Conv2d(3*3, 9, kernel_size=KERNEL, padding=PADDING),
                                       nn.InstanceNorm2d(9),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(3 * 3, 3, kernel_size=KERNEL, padding=PADDING),)


    def reparameterize(self, mu, logvar,gate):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp*gate
        return z

    def bottleneck(self, h,gate):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar,gate)
        return z, mu, logvar

    def bottleneckdown2(self, h,gate):
        mu, logvar = self.down2fc1(h), self.down2fc2(h)
        z = self.reparameterize(mu, logvar,gate)
        return z, mu, logvar

    def bottleneckdown4(self, h,gate):
        mu, logvar = self.down4fc1(h), self.down4fc2(h)
        z = self.reparameterize(mu, logvar,gate)
        return z, mu, logvar

    def encode(self, x,gate):
        out1 = self.conv_seq1(x)
        out2 = self.conv_seq2(self.maxpool(out1))
        out3 = self.conv_seq3(self.maxpool(out2))
        out4 = self.conv_seq4(self.maxpool(out3))
        out5 = self.conv_seq5(self.maxpool(out4))

        deout1 = self.deconv_seq1(torch.cat((self.convt1(out5),out4),1))
        deout2 = self.deconv_seq2(torch.cat((self.convt2(deout1),out3),1))
        feat_down4,down4_mu,down4_logvar = self.bottleneckdown4(deout2,gate)
        segout_down4 = self.segdown4_seq(feat_down4)
        pred_down4 = self.soft(segout_down4)
        deout3 = self.deconv_seq3(torch.cat((self.convt3(feat_down4),out2),1))
        feat_down2,down2_mu,down2_logvar = self.bottleneckdown2(deout3,gate)
        segout_down2 = self.segdown2_seq(feat_down2)
        pred_down2 = self.soft(segout_down2)
        deout4 = self.deconv_seq4(torch.cat((self.convt4(feat_down2),out1),1))
        z, mu, logvar = self.bottleneck(deout4,gate)
        return z, mu, logvar,pred_down2,segout_down2,feat_down2,down2_mu,down2_logvar,pred_down4,segout_down4,feat_down4,down4_mu,down4_logvar,out5


    def forward(self, x,gate):
        z, mu, logvar,pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar,pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar,out5 = self.encode(x,gate)
        out= self.deconv_seq5(z)
        pred = self.soft(out)
        fusion_seg = self.segfusion(torch.cat((pred,self.upsample2(pred_down2),self.upsample4(pred_down4)),dim=1))

        return fusion_seg,pred,out,z, mu, logvar,pred_down2, segout_down2, feat_down2, down2_mu, down2_logvar,pred_down4, segout_down4, feat_down4, down4_mu, down4_logvar,out5



class BasicBlock_Change(nn.Module):
    def __init__(self, inplanes, planes,kernel_size=3, padding=1,dilation=1):
        super(BasicBlock_Change, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,padding=padding,dilation=dilation, bias=True)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,kernel_size=kernel_size,padding=padding,dilation=dilation, bias=True)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes,kernel_size=kernel_size,padding=padding,dilation=dilation, bias=True),
                                        nn.InstanceNorm2d(planes),
                                        )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_NoChange(nn.Module):
    def __init__(self, inplanes,kernel_size=3,padding=1,dilation=1):
        super(BasicBlock_NoChange, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, padding=padding,dilation=dilation, bias=True)
        self.bn1 = nn.InstanceNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, padding=padding,dilation=dilation, bias=True)
        self.bn2 = nn.InstanceNorm2d(inplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self,fb=16 ):
        super(Decoder, self).__init__()
        self.ResBlock1 = BasicBlock_NoChange(fb * 4, kernel_size=3, padding=1, dilation=1)
        self.ResBlock2 = BasicBlock_NoChange(fb * 4, kernel_size=3, padding=1, dilation=1)

        self.ResBlock3 = BasicBlock_Change(fb * 4,fb * 2, kernel_size=5, padding=4, dilation=2)
        self.ResBlock4 = BasicBlock_NoChange(fb * 2, kernel_size=5, padding=4, dilation=2)

        self.conv_seq1 = nn.Sequential(nn.Conv2d(fb * 2,64,kernel_size=3, padding=1),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True))

        self.conv_seq2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3, padding=1),
                                       nn.InstanceNorm2d(64),
                                       nn.ReLU(inplace=True))

        self.conv_seq3 = nn.Sequential(nn.Conv2d(64,16,kernel_size=3, padding=1),
                                       nn.InstanceNorm2d(16),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(16, 1, kernel_size=3, padding=1),
                                       nn.Sigmoid(),
                                       )




    def forward(self, x):
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, KERNEL=3, PADDING=1):
        super(Discriminator, self).__init__()

        self.decoder=nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2),  # 190
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, stride=2),  # 190
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, dilation=2),  # 190
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2,dilation=2),  # 190
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),  # (190-3)/2+1=94
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.linear_seq=nn.Sequential(nn.Linear(32*5*5,256),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(256, 64),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(64, 1),
                                      )

    def forward(self, y):
        out= self.decoder(y)
        out = self.linear_seq(out.view(out.size(0),-1))
        out = out.mean()
        return out


class DataSet_Train(Dataset):
    def __init__(self, list):
        self.img = list
        self.shape = [220, 240]
        self.size = 192

    def __getitem__(self, item):
        imgcase, randcase = divmod(item, 4)
        randx = np.random.randint(0, self.shape[0] - self.size)
        randy = np.random.randint(0, self.shape[1] - self.size)
        imagename = self.img[imgcase]
        labelname = imagename.replace('img', 'lab')

        itkimg = sitk.ReadImage(imagename)
        npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
        npimg = np.squeeze(npimg)
        npimg = npimg.astype(np.float32)

        # read label data
        itklab = sitk.ReadImage(labelname)
        nplab = sitk.GetArrayFromImage(itklab)
        nplab = np.squeeze(nplab)

        nplabs = (nplab == 205) * 1 + (nplab == 500) * 2
        nplabs = nplabs[randx:randx + self.size, randy:randy + self.size]
        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        npimg = npimg.astype(np.float32)[randx:randx + self.size, randy:randy + self.size]

        npimg_down2 = transform.resize(npimg, (self.size//2, self.size//2), order=3, mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (self.size//4, self.size//4), order=3, mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(nplabs, (self.size//2, self.size//2), order=0, mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(nplabs, (self.size//4, self.size//4), order=0, mode='edge', preserve_range=True)

        return torch.from_numpy(np.expand_dims(npimg,0)),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(nplabs),torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor)


    def __len__(self):
        size = len(self.img) * 4
        return size

class DataSet_Test(Dataset):
    def __init__(self, path):
        self.dir = path
        self.img = glob.glob(path + '/img*')

    def __getitem__(self, item):
        imagename = self.img[item]
        labelname = imagename.replace('img', 'lab')

        itkimg = sitk.ReadImage(imagename)
        npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X
        npimg = np.squeeze(npimg)
        npimg = npimg.astype(np.float32)

        # read label data
        itklab = sitk.ReadImage(labelname)
        nplab = sitk.GetArrayFromImage(itklab)
        nplab = np.squeeze(nplab)

        npimg = npimg[14:206, :192]
        nplab = nplab[14:206, :192]

        nplabs = (nplab == 205) * 1 + (nplab == 500) * 2
        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        npimg = npimg.astype(np.float32)

        filename = labelname.split("/")[-1]

        return torch.from_numpy(np.expand_dims(npimg, 0)), torch.from_numpy(nplabs), filename[:-7]

    def __len__(self):
        return len(self.img)

class LGE_TrainSet0(Dataset):
    def __init__(self):
        self.imgdir='/Patch192/LGE/'

        self.imgsname = glob.glob(self.imgdir + '*LGE.nii*')

        imgs = np.zeros((1,192,192))
        self.info = []
        for img_num in range(len(self.imgsname)):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)

            imgs = np.concatenate((imgs,npimg),axis=0)
            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:,:,:]

    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,4)

        npimg = self.imgs[imgindex,:,:]
        randx = np.random.randint(-32,32)
        randy = np.random.randint(-32, 32)
        npimg=npimg[96+randx-64:96+randx+64,96+randy-64:96+randy+64]

        # npimg_o = transform.resize(npimg, (80, 80),
        #                      order=3, mode='edge', preserve_range=True)
        #npimg_resize = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        npimg_down2 = transform.resize(npimg, (64,64 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (32,32 ), order=3,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):

        return self.imgs.shape[0]*4



class C0_TrainSet0(Dataset):
    def __init__(self):
        self.imgdir = '/Patch192/C0/'

        self.imgsname = glob.glob(self.imgdir + '*C0.nii*')

        imgs = np.zeros((1,192,192))
        labs = np.zeros((1,192,192))
        self.info = []
        for img_num in range(len(self.imgsname)):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            imgs = np.concatenate((imgs,npimg),axis=0)

            labname = self.imgsname[img_num].replace('.nii','_manual.nii')
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

            spacing = itkimg.GetSpacing()[2]
            media_slice = int(npimg.shape[0] / 2)
            for i in range(npimg.shape[0]):
                a, _ = divmod((i - media_slice) * spacing, 20.0)
                info = int(a) + 3
                if info < 0:
                    info = 0
                elif info > 5:
                    info = 5

                self.info.append(info)
        self.imgs = imgs[1:,:,:]
        self.labs = labs[1:,:,:]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)



    def __getitem__(self, item):
        imgindex,crop_indice = divmod(item,4)

        npimg = self.imgs[imgindex,:,:]
        nplab = self.labs[imgindex,:,:]

        # npimg = transform.resize(npimg, (96, 96), order=3,mode='edge', preserve_range=True)
        # nplab = transform.resize(nplab, (96, 96), order=0,mode='edge', preserve_range=True)
        randx = np.random.randint(-32,32)
        randy = np.random.randint(-32, 32)
        npimg=npimg[96+randx-64:96+randx+64,96+randy-64:96+randy+64]
        nplab=nplab[96+randx-64:96+randx+64,96+randy-64:96+randy+64]

        # npimg_o=transform.resize(npimg, (80,80 ), order=3,mode='edge', preserve_range=True)
        # nplab_o=transform.resize(nplab, (80,80 ), order=0,mode='edge', preserve_range=True)

        npimg_down2 = transform.resize(npimg, (64,64 ), order=3,mode='edge', preserve_range=True)
        npimg_down4 = transform.resize(npimg, (32,32 ), order=3,mode='edge', preserve_range=True)

        nplab_down2 = transform.resize(nplab, (64,64 ), order=0,mode='edge', preserve_range=True)
        nplab_down4 = transform.resize(nplab, (32,32), order=0,mode='edge', preserve_range=True)

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down2).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(npimg_down4).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(nplab).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down2).type(dtype=torch.LongTensor),torch.from_numpy(nplab_down4).type(dtype=torch.LongTensor),torch.tensor(self.info[imgindex]).type(dtype=torch.LongTensor)

    def __len__(self):

        return self.imgs.shape[0]*4




def dice_compute(pred, groundtruth):           #batchsize*channel*W*W
    # for j in range(pred.shape[0]):
    #     for i in range(pred.shape[1]):
    #         if np.sum(pred[j,i,:,:])==0 and np.sum(groundtruth[j,i,:,:])==0:
    #             pred[j, i, :, :]=pred[j, i, :, :]+1
    #             groundtruth[j, i, :, :]=groundtruth[j,i,:,:]+1
    #
    # dice = 2*np.sum(pred*groundtruth,axis=(2,3),dtype=np.float16)/(np.sum(pred,axis=(2,3),dtype=np.float16)+np.sum(groundtruth,axis=(2,3),dtype=np.float16))
    dice=[]
    for i in range(3):
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]


    return np.array(dice,dtype=np.float32)




def IOU_compute(pred, groundtruth):
    iou=[]
    for i in range(3):
        iou_i = (np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)-np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)
        iou=iou+[iou_i]


    return np.array(iou,dtype=np.float32)


def Hausdorff_compute(pred,groundtruth,spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1,3, 5))
    surface_distance_results = np.zeros((1,3, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(3):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0 or np.sum(groundtruth==i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)


    return overlap_results[0,:,1],surface_distance_results[0,:,1]

def multi_dice_iou_compute(pred,label):
    truemax, truearg = torch.max(pred, 1, keepdim=False)
    truearg = truearg.detach().cpu().numpy()
    # nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, \
    #                    truearg == 4, truearg == 5, truearg == 6, truearg == 7), 1)
    nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, truearg == 4, truearg == 5), 1)
    # truelabel = (truearg == 0) * 550 + (truearg == 1) * 420 + (truearg == 2) * 600 + (truearg == 3) * 500 + \
    #             (truearg == 4) * 250 + (truearg == 5) * 850 + (truearg == 6) * 820 + (truearg == 7) * 0

    dice = dice_compute(nplabs, label.cpu().numpy())
    Iou = IOU_compute(nplabs, label.cpu().numpy())

    return dice,Iou


class BalancedBCELoss(nn.Module):
    def __init__(self,target):
        super(BalancedBCELoss,self).__init__()
        self.eps=1e-6
        weight = torch.tensor([torch.reciprocal(torch.sum(target==0).float()+self.eps),torch.reciprocal(torch.sum(target==1).float()+self.eps),torch.reciprocal(torch.sum(target==2).float()+self.eps)])
        self.criterion = nn.CrossEntropyLoss(weight)

    def forward(self, output,target):
        loss = self.criterion(output,target)

        return loss



class Gaussian_Kernel_Function(nn.Module):
    def __init__(self,std):
        super(Gaussian_Kernel_Function, self).__init__()
        self.sigma=std**2

    def forward(self, fa,fb):
        asize = fa.size()
        bsize = fb.size()

        fa1 = fa.view(-1, 1, asize[1])
        fa2 = fa.view(1, -1, asize[1])

        fb1 = fb.view(-1, 1, bsize[1])
        fb2 = fb.view(1, -1, bsize[1])

        aa = fa1-fa2
        vaa = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(aa,2,dim=2),2),self.sigma)))

        bb = fb1-fb2
        vbb = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(bb,2,dim=2),2),self.sigma)))

        ab = fa1-fb2
        vab = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(ab,2,dim=2),2),self.sigma)))

        loss = vaa+vbb-2.0*vab

        return loss




class Feature_Distribution_Distance_Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature_modala, feature_modalb, Tvalue):
        ctx.save_for_backward(feature_modala, feature_modalb)
        ctx.Tvalue = Tvalue
        ctx.asize = feature_modala.size()
        ctx.bsize = feature_modalb.size()
        out_channel = ctx.asize[1]
        feature_modala1 = feature_modala.view(-1, 1, out_channel)
        feature_modala2 = feature_modala.view(1, -1, out_channel)

        feature_modalb1 = feature_modalb.view(-1, 1, out_channel)
        feature_modalb2 = feature_modalb.view(1, -1, out_channel)

        ctx.matrix_xx = feature_modala1 - feature_modala2
        ctx.index_zero_xx = (ctx.matrix_xx == 0).type(torch.cuda.FloatTensor)
        dmatrix_xx = torch.mean(
            torch.div((1 - ctx.index_zero_xx) * torch.sin(ctx.matrix_xx * Tvalue) + Tvalue * ctx.index_zero_xx,
                      (1 - ctx.index_zero_xx) * ctx.matrix_xx + ctx.index_zero_xx))

        ctx.matrix_xy = feature_modala1 - feature_modalb2
        ctx.index_zero_xy = (ctx.matrix_xy == 0).type(torch.cuda.FloatTensor)
        dmatrix_xy = torch.mean(
            torch.div((1 - ctx.index_zero_xy) * torch.sin(ctx.matrix_xy * Tvalue) + Tvalue * ctx.index_zero_xy,
                      (1 - ctx.index_zero_xy) * ctx.matrix_xy + ctx.index_zero_xy))

        ctx.matrix_yy = feature_modalb1 - feature_modalb2
        ctx.index_zero_yy = (ctx.matrix_yy == 0).type(torch.cuda.FloatTensor)
        dmatrix_yy = torch.mean(
            torch.div((1 - ctx.index_zero_yy) * torch.sin(ctx.matrix_yy * Tvalue) + Tvalue * ctx.index_zero_yy,
                      (1 - ctx.index_zero_yy) * ctx.matrix_yy + ctx.index_zero_yy))

        loss = dmatrix_xx + dmatrix_yy - 2.0 * dmatrix_xy

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # feature_modala, feature_modalb = ctx.saved_variables
        grad_feature_modala = grad_feature_modalb = None

        derix_dmatrix_xx = torch.sum(2 * torch.div(
            ctx.Tvalue * torch.cos(ctx.matrix_xx * ctx.Tvalue) * ctx.matrix_xx - torch.sin(
                ctx.matrix_xx * ctx.Tvalue), (1 - ctx.index_zero_xx) * (ctx.matrix_xx ** 2) + ctx.index_zero_xx),
                                     dim=1) / (ctx.asize[0] * ctx.asize[0] * ctx.asize[1])
        deriy_dmatrix_yy = torch.sum(2 * torch.div(
            ctx.Tvalue * torch.cos(ctx.matrix_yy * ctx.Tvalue) * ctx.matrix_yy - torch.sin(
                ctx.matrix_yy * ctx.Tvalue), (1 - ctx.index_zero_yy) * (ctx.matrix_yy ** 2) + ctx.index_zero_yy),
                                     dim=1) / (ctx.bsize[0] * ctx.bsize[0] * ctx.bsize[1])
        derix_dmatrix_xy = torch.sum(torch.div(
            ctx.Tvalue * torch.cos(ctx.matrix_xy * ctx.Tvalue) * ctx.matrix_xy - torch.sin(
                ctx.matrix_xy * ctx.Tvalue), (1 - ctx.index_zero_xy) * (ctx.matrix_xy ** 2) + ctx.index_zero_xy),
                                     dim=1) / (ctx.asize[0] * ctx.bsize[0] * ctx.asize[1])
        deriy_dmatrix_xy = torch.sum(torch.div(
            -ctx.Tvalue * torch.cos(ctx.matrix_xy * ctx.Tvalue) * ctx.matrix_xy + torch.sin(
                ctx.matrix_xy * ctx.Tvalue), (1 - ctx.index_zero_xy) * (ctx.matrix_xy ** 2) + ctx.index_zero_xy),
                                     dim=0) / (ctx.asize[0] * ctx.bsize[0] * ctx.asize[1])

        if ctx.needs_input_grad[0]:
            grad_feature_modala = grad_output * (derix_dmatrix_xx - 2.0 * derix_dmatrix_xy)
            grad_feature_modala = grad_feature_modala.view(ctx.asize)
            # print(grad_feature_modala.max().item())

        if ctx.needs_input_grad[1]:
            grad_feature_modalb = grad_output * (deriy_dmatrix_yy - 2.0 * deriy_dmatrix_xy)
            grad_feature_modalb = grad_feature_modalb.view(ctx.bsize)

        return grad_feature_modala, grad_feature_modalb, None

class Feature_Distribution_Distance_func(nn.Module):
    def __init__(self, avg_kernel, avg_stride):
        super(Feature_Distribution_Distance_func, self).__init__()
        # self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernelsize, bias=False)
        self.avgpool = nn.AvgPool2d(avg_kernel, avg_stride)
        # self.out_channel = out_channel
        self.weightclip = [-1.0, 1.0]
        self.Tvalue = 1e2  # 1000000.0 #torch.tensor(1e6).type(torch.cuda.HalfTensor)

    def forward(self, feature_modala, feature_modalb):
        feature_modala = self.avgpool(feature_modala)
        feature_modalb = self.avgpool(feature_modalb)

        asize = feature_modala.size()
        bsize = feature_modalb.size()

        feature_modala = feature_modala.view(asize[0], -1, 1, 1)
        feature_modalb = feature_modalb.view(bsize[0], -1, 1, 1)

        loss = Feature_Distribution_Distance_Func.apply(feature_modala, feature_modalb, self.Tvalue)

        return loss

# class Gaussian_Distance(nn.Module):
#     def __init__(self):
#         super(Gaussian_Distance, self).__init__()
#         self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
#
#     def forward(self, mu_a,logvar_a,mu_b,logvar_b):
#         mu_a = self.avgpool(mu_a)
#         mu_b = self.avgpool(mu_b)
#         # var_a = torch.exp(logvar_a)
#         # var_b = torch.exp(logvar_b)
#         var_a = self.avgpool(torch.exp(logvar_a))/4
#         var_b = self.avgpool(torch.exp(logvar_b))/4
#
#
#         mu_a1 = mu_a.view(-1,1,mu_a.size(1))
#         mu_a2 = mu_a.view(1,-1,mu_a.size(1))
#         var_a1 = var_a.view(-1,1,var_a.size(1))
#         var_a2 = var_a.view(1,-1,var_a.size(1))
#
#         mu_b1 = mu_b.view(-1,1,mu_b.size(1))
#         mu_b2 = mu_b.view(1,-1,mu_b.size(1))
#         var_b1 = var_b.view(-1,1,var_b.size(1))
#         var_b2 = var_b.view(1,-1,var_b.size(1))
#
#         vaa = torch.mean(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_a2,2),var_a1+var_a2),-0.5)),torch.sqrt(var_a1+var_a2)))
#         vab = torch.mean(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_a1-mu_b2,2),var_a1+var_b2),-0.5)),torch.sqrt(var_a1+var_b2)))
#         vbb = torch.mean(torch.div(torch.exp(torch.mul(torch.div(torch.pow(mu_b1-mu_b2,2),var_b1+var_b2),-0.5)),torch.sqrt(var_b1+var_b2)))
#
#         # vaa = torch.mean((mu_a1-mu_a2).pow_(2).div_(var_a1+var_a2).mul_(-0.5).exp_().div_((var_a1+var_a2).sqrt_()))
#         # vab = torch.mean((mu_a1-mu_b2).pow_(2).div_(var_a1+var_b2).mul_(-0.5).exp_().div_((var_a1+var_b2).sqrt_()))
#         # vbb = torch.mean((mu_b1-mu_b2).pow_(2).div_(var_b1+var_b2).mul_(-0.5).exp_().div_((var_b1+var_b2).sqrt_()))
#
#         loss = vaa+vbb-torch.mul(vab,2.0)
#
#         return loss

