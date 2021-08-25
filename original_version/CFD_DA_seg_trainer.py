import torch
from torch import nn
from torch.utils.data import Dataset
import os
import math
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import time
import scipy.misc
from utils_for_transfer import *

EPOCH = 25
KLDLamda=1.0

# PredLamda=1e3
# DisLamda=1e-4
LR = 1e-4
ADA_DisLR=1e-4

WEIGHT_DECAY =1e-5
WORKERSNUM = 20
#TestDir=['/home/wfp/2019TMI/LGE_C0_T2/Original/c0t2lgeCropNorm/LGE192_Validation/','/home/wfp/2019TMI/LGE_C0_T2/Original/c0t2lgeCropNorm/LGE192/']
TestDir=['/home/wfp/2019TMI/LGE_C0_T2/Original/Patch192/LGE_Test/','/home/wfp/2019TMI/LGE_C0_T2/Original/Patch192/LGE_Vali/']
BatchSize = 10
KERNEL=1
CT_Train_Dir = '/home/wfp/CVPR2018/data/train_validate_data/2Dslice/ct_train'
MR_Train_Dir='/home/wfp/CVPR2018/data/train_validate_data/2Dslice/mr_train'
MR_Test_Dir = '/home/wfp/CVPR2018/data/train_validate_data/2Dslice/mr_validate'
prefix='/home/wfp/2019CFD/TMI_Review/R1_parameter/CFD_U3'
# SAVE_DIR =prefix+ '/save_train_param'
# SAVE_IMG_DIR=prefix+'/save_test_label'

def ADA_Train(save_loss_txt,save_cfd_txt, Train_LoaderA,Train_LoaderB,encoder,gate,DistanceNet,lr,predlamda,dislamda,meanlamda,epoch,optim, savedir):
    lr=lr*(0.9**(epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr


    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)

    i=0


    while i<len(A_iter) and i<len(B_iter):
        ct,ct_down2,ct_down4,label,label_down2,label_down4 = A_iter.next()
        mr,mr_down2,mr_down4,_,_,_= B_iter.next()

        ct= ct.cuda()
        ct_down2= ct_down2.cuda()
        ct_down4= ct_down4.cuda()


        mr= mr.cuda()
        mr_down4= mr_down4.cuda()
        mr_down2= mr_down2.cuda()


        label= label.cuda()

        label_down2= label_down2.cuda()

        label_down4= label_down4.cuda()

        fusionseg,_, out_ct,feat_ct, mu_ct,logvar_ct, _, outdown2_ct,featdown2_ct, mudown2_ct,logvardown2_ct,_, outdown4_ct,featdown4_ct, mudown4_ct,logvardown4_ct,info_pred_ct= encoder(ct,gate)


        seg_criterian = BalancedBCELoss(label)
        seg_criterian = seg_criterian.cuda()
        segloss_output = seg_criterian(out_ct, label)
        fusionsegloss_output = seg_criterian(fusionseg, label)

        segdown2_criterian = BalancedBCELoss(label_down2)
        segdown2_criterian = segdown2_criterian.cuda()
        segdown2loss_output = segdown2_criterian(outdown2_ct, label_down2)

        segdown4_criterian = BalancedBCELoss(label_down4)
        segdown4_criterian = segdown4_criterian.cuda()
        segdown4loss_output = segdown4_criterian(outdown4_ct, label_down4)


        _,pred_mr, _,feat_mr, mu_mr,logvar_mr, preddown2_mr, _,featdown2_mr, mudown2_mr,logvardown2_mr,preddown4_mr, _,featdown4_mr, mudown4_mr,logvardown4_mr,info_pred_mr= encoder(mr,gate)

        distance_loss = DistanceNet(feat_ct,feat_mr)
        distance_down2_loss = DistanceNet(featdown2_ct,featdown2_mr)
        distance_down4_loss = DistanceNet(featdown4_ct,featdown4_mr)

        meanloss1 = torch.mean((feat_ct.mean(dim=0, keepdim=True) - feat_mr.mean(dim=0, keepdim=True)) ** 2)
        meanloss2 = torch.mean((featdown2_ct.mean(dim=0, keepdim=True) - featdown2_mr.mean(dim=0, keepdim=True)) ** 2)
        meanloss3 = torch.mean((featdown4_ct.mean(dim=0, keepdim=True) - featdown4_mr.mean(dim=0, keepdim=True)) ** 2)



        balanced_loss = dislamda*(distance_loss+distance_down2_loss+distance_down4_loss)+predlamda*(segloss_output+fusionsegloss_output+segdown2loss_output+segdown4loss_output)+ meanlamda*(meanloss1+meanloss2+meanloss3)

        optim.zero_grad()
        balanced_loss.backward()
        optim.step()

        f = open(save_loss_txt, 'a')
        f.write('{0:.4f}\n'.format(balanced_loss))
        f.close()

        f = open(save_cfd_txt, 'a')
        f.write('{0:.4f}\n'.format(distance_loss + distance_down2_loss + distance_down4_loss))
        f.close()

        if i % 20 == 0:
            print('epoch %d , %d th iter; seglr,ADA_totalloss,segloss,distance_loss1,distance_loss2: %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'\
                  % (epoch, i,lr, balanced_loss.item(),fusionsegloss_output.item(),segloss_output.item(),segdown2loss_output.item(),segdown4loss_output.item(),distance_loss.item(),distance_down2_loss.item(),distance_down4_loss.item()))

        i=i+1

def SegNet_test_mr(mrtest_loader, mrSegNet, epoch,ePOCH, save_DIR,save_IMG_DIR):
    total_loss = 0.0
    total_dice = np.zeros((3,))
    total_Iou = np.zeros((3,))
    total_avghausdorff = np.zeros((3,))
    total_hausdorff = np.zeros((3,))
    total_itkdice = np.zeros((3,))
    num = 0
    mrSegNet.eval()
    for i,(data,label,filename) in enumerate(mrtest_loader):
        data = data.cuda()
        label = label.cuda()
        output,_,_, _, _,_,_, _,_,_,_, _, _,_,_, _,_ = mrSegNet(data,0)
        loss = BalancedBCELoss(label)
        loss = loss.cuda()
        pred_loss = loss(output,label)
        total_loss+=pred_loss.item()

        truemax, truearg = torch.max(output, 1, keepdim=False)
        truearg = truearg.detach().cpu().numpy()
        if epoch==ePOCH-1:

            truelabel = (truearg == 1) * 205  + (truearg == 2) * 500 # \
                       # (truearg == 4) * 250 + (truearg == 5) * 850 + (truearg == 6) * 820 + (truearg == 7) * 0
            scipy.misc.imsave('%s/mr_%s_testout.jpg'%(save_IMG_DIR,filename[0]), np.squeeze(truelabel))
        dice = dice_compute(truearg,label.cpu().numpy())
        Iou = IOU_compute(truearg,label.cpu().numpy())
        itkavg_hausdorff, itk_hausdorff, itk_dice = Hausdorff_compute(truearg,label.cpu().numpy())

        total_dice = np.vstack((total_dice,dice))
        total_Iou = np.vstack((total_Iou,Iou))
        total_avghausdorff = np.vstack((total_avghausdorff,itkavg_hausdorff))
        total_hausdorff = np.vstack((total_hausdorff,itk_hausdorff))
        total_itkdice = np.vstack((total_itkdice,itk_dice))

        num+=1

    if num==0:
        return
    else:
        meanloss = total_loss/num

        meanDice = np.mean(total_dice[1:],axis=0)
        stdDice = np.std(total_dice[1:],axis=0)

        meanIou = np.mean(total_Iou[1:],axis=0)
        stdIou = np.std(total_Iou[1:],axis=0)

        mean_avghausdorff = np.mean(total_avghausdorff[1:], axis=0)
        std_avghausdorff = np.std(total_avghausdorff[1:], axis=0)

        mean_hausdorff = np.mean(total_hausdorff[1:], axis=0)
        std_hausdorff = np.std(total_hausdorff[1:], axis=0)

        mean_itkdice = np.mean(total_itkdice[1:], axis=0)
        std_itkdice = np.std(total_itkdice[1:], axis=0)

        with open("%s/mr_testout_index.txt" % save_DIR, "a") as f:
            f.writelines(["\nepoch:", str(epoch), " ", "meanloss:", str(meanloss),"","meanDice:",""\
                             ,str(meanDice.tolist()),"stdDice:","",str(stdDice.tolist()),"","meanIou:","",str(meanIou.tolist()),"stdIou:","",str(stdIou.tolist()), \
                              "", "meanavghausdorff:", "", str(mean_avghausdorff.tolist()), "stdavghausdorff:", "", str(std_avghausdorff.tolist()), \
                              "", "meanhausdorff:", "", str(mean_hausdorff.tolist()), "stdhausdorff:", "", str(std_hausdorff.tolist()), \
                              "", "meanitkdice:", "", str(mean_itkdice.tolist()), "stditkdice:", "", str(std_itkdice.tolist())])




def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    cudnn.benchmark = True
    vaeencoder = VAE()
    vaeencoder = vaeencoder.cuda()



    DistanceNet = Feature_Distribution_Distance_func(KERNEL,KERNEL)  #64,Num_Feature2,(12,12)
    DistanceNet = DistanceNet.cuda()
    #DistanceNet2 = nn.DataParallel(DistanceNet2, device_ids=[0,1])


    DA_optim = torch.optim.Adam([{'params': vaeencoder.parameters()}],lr=LR,weight_decay=WEIGHT_DECAY)


    SourceData = DataSet_Train(CT_Train_Dir)
    SourceData_loader = DataLoader(SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)

    TargetData = DataSet_Train(MR_Train_Dir)
    TargetData_loader = DataLoader(TargetData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)


    MR_Testingdata = DataSet_Test(MR_Test_Dir)
    MRtest_loader = DataLoader(MR_Testingdata, batch_size=1, shuffle=False, num_workers=WORKERSNUM,pin_memory=True)
    # TestData = LabeledDataSet(modality='mr',stage='test')
    # TestData_loader = DataLoader(TestData, batch_size=1, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)
    PredLamda=1e3
    DisLamdaList=[0.0]
    DisLamdaListDown2=[1e-4,1e-5]
    DisLamdaListDown4=[1e-4,1e-5]
    MeanList=[1e3]


    for MeanLamda in MeanList:
        for DisLamda in DisLamdaList:

            print ('PredLamda'+str(PredLamda)+',  DisLamda'+str(DisLamda)+',  DisLamdaDown4'+str(MeanLamda))
            print ('\n')
            print ('\n')
            SAVE_DIR=prefix+'/save_train_param'+'_DisLamda'#+str(int(round(math.log(DisLamda,10))))+str(int(round(math.log(MeanLamda,10))))
            SAVE_IMG_DIR=prefix+'/save_test_label'+'_DisLamda'#+str(int(round(math.log(DisLamda,10))))+str(int(round(math.log(MeanLamda,10))))
            if not os.path.exists(SAVE_DIR):
                os.mkdir(SAVE_DIR)
            if not os.path.exists(SAVE_IMG_DIR):
                os.mkdir(SAVE_IMG_DIR)
            vaeencoder.apply(init_weights)

            save_loss_txt = SAVE_DIR + '/loss_iter.txt'
            save_cfd_txt = SAVE_DIR + '/cfdloss_iter.txt'

            for epoch in range(EPOCH):
                vaeencoder.train()
                ADA_Train(save_loss_txt,save_cfd_txt, SourceData_loader,TargetData_loader,vaeencoder,0.0,DistanceNet,LR,PredLamda,DisLamda,MeanLamda,epoch,DA_optim, SAVE_DIR)
                vaeencoder.eval()
                SegNet_test_mr(MRtest_loader, vaeencoder, epoch, EPOCH, SAVE_DIR, SAVE_IMG_DIR)

                if epoch==EPOCH-1:
                    torch.save(vaeencoder.state_dict(), os.path.join(SAVE_DIR, 'encoder_param.pkl'))





if __name__ == '__main__':
    main()
