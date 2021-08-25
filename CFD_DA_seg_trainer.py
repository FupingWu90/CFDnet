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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import argparse


def ADA_Train( Train_LoaderA,Train_LoaderB,encoder,decoderB,decoderBdown2,decoderBdown4,gate,DistanceNet,lr,kldlamda,predlamda,bcelamda,dislamda,epoch,optim, savedir):
    lr=lr*(0.9**(epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr


    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)

    i=0


    while i<len(A_iter) and i<len(B_iter):
        optim.zero_grad()
        ct,ct_down2,ct_down4,label,label_down2,label_down4 = A_iter.next()
        mr,mr_down2,mr_down4,_,_,_= B_iter.next()

        ct= ct.cuda()
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

        KLD_ct = -0.5 * torch.mean(1 + logvar_ct - mu_ct.pow(2) - logvar_ct.exp())
        KLD_down2_ct = -0.5 * torch.mean(1 + logvardown2_ct - mudown2_ct.pow(2) - logvardown2_ct.exp())
        KLD_down4_ct = -0.5 * torch.mean(1 + logvardown4_ct - mudown4_ct.pow(2) - logvardown4_ct.exp())

        # loss_ct = predlamda*(segloss_output+fusionsegloss_output+segdown2loss_output+segdown4loss_output)+kldlamda*(KLD_ct+KLD_down2_ct+KLD_down4_ct)
        # loss_ct.backward(retain_graph=True)

        _,pred_mr, _,feat_mr, mu_mr,logvar_mr, preddown2_mr, _,featdown2_mr, mudown2_mr,logvardown2_mr,preddown4_mr, _,featdown4_mr, mudown4_mr,logvardown4_mr,info_pred_mr= encoder(mr,gate)


        recon_mr=decoderB(feat_mr)
        BCE_mr = F.binary_cross_entropy(recon_mr, mr)
        KLD_mr = -0.5 * torch.mean(1 + logvar_mr - mu_mr.pow(2) - logvar_mr.exp())

        recondown2_mr=decoderBdown2(featdown2_mr)
        BCE_down2_mr = F.binary_cross_entropy(recondown2_mr, mr_down2)
        KLD_down2_mr = -0.5 * torch.mean(1 + logvardown2_mr - mudown2_mr.pow(2) - logvardown2_mr.exp())

        recondown4_mr=decoderBdown4(featdown4_mr)
        BCE_down4_mr = F.binary_cross_entropy(recondown4_mr, mr_down4)
        KLD_down4_mr = -0.5 * torch.mean(1 + logvardown4_mr - mudown4_mr.pow(2) - logvardown4_mr.exp())

        # loss_mr = kldlamda*(KLD_mr+KLD_down2_mr+KLD_down4_mr)+bcelamda*(BCE_mr+BCE_down2_mr+BCE_down4_mr)
        # loss_mr.backward(retain_graph=True)

        distance_loss = DistanceNet(feat_ct,feat_mr)
        distance_down2_loss = DistanceNet(featdown2_ct,featdown2_mr)
        distance_down4_loss = DistanceNet(featdown4_ct,featdown4_mr)

        meanloss1 = torch.mean((feat_ct.mean(dim=0, keepdim=True) - feat_mr.mean(dim=0, keepdim=True)) ** 2)
        meanloss2 = torch.mean((featdown2_ct.mean(dim=0, keepdim=True) - featdown2_mr.mean(dim=0, keepdim=True)) ** 2)
        meanloss3 = torch.mean((featdown4_ct.mean(dim=0, keepdim=True) - featdown4_mr.mean(dim=0, keepdim=True)) ** 2)

       # balanced_loss = predlamda*(segloss_output+fusionsegloss_output+segdown2loss_output+segdown4loss_output)+kldlamda*(KLD_ct+KLD_down2_ct+KLD_down4_ct+KLD_mr+KLD_down2_mr+KLD_down4_mr)+ 1e3*(meanloss1+meanloss2+meanloss3)+dislamda*(distance_loss+distance_down2_loss+distance_down4_loss)+bcelamda*(BCE_mr+BCE_down2_mr+BCE_down4_mr)
       #  balanced_loss = predlamda *segloss_output + predlamda *fusionsegloss_output + predlamda *segdown2loss_output + predlamda *segdown4loss_output + kldlamda * KLD_ct + kldlamda *KLD_down2_ct +kldlamda * KLD_down4_ct + kldlamda *KLD_mr + kldlamda *KLD_down2_mr + kldlamda *KLD_down4_mr + 1e3 * (
       #  meanloss1 + meanloss2 + meanloss3) + dislamda * distance_loss + dislamda *distance_down2_loss + dislamda *distance_down4_loss + bcelamda * BCE_mr + bcelamda *BCE_down2_mr + bcelamda *BCE_down4_mr
        balanced_loss = bcelamda * BCE_mr + torch.mul(KLD_mr, kldlamda) + torch.mul(KLD_ct, kldlamda) + torch.mul(distance_loss,
                                                                                                      dislamda) + predlamda * (
        segloss_output + fusionsegloss_output) + \
                        torch.mul(KLD_down2_ct, kldlamda) + bcelamda * BCE_down2_mr + torch.mul(KLD_down2_mr,
                                                                                          kldlamda) + torch.mul(
            distance_down2_loss, dislamda) + predlamda * segdown2loss_output + \
                        torch.mul(KLD_down4_ct, kldlamda) + bcelamda * BCE_down4_mr + torch.mul(KLD_down4_mr,
                                                                                          kldlamda) + torch.mul(
            distance_down4_loss, dislamda) + predlamda * segdown4loss_output + 1e3 * (
        meanloss1 + meanloss2 + meanloss3)

        balanced_loss.backward()
        optim.step()


        if i % 20 == 0:
            print('epoch %d , %d th iter; seglr,ADA_totalloss,segloss,distance_loss1,distance_loss2: %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'\
                  % (epoch, i,lr, balanced_loss.item(),fusionsegloss_output.item(),segloss_output.item(),segdown2loss_output.item(),segdown4loss_output.item(),distance_loss.item(),distance_down2_loss.item(),distance_down4_loss.item()))

        i=i+1

def SegNet_test_mr(testfile_list, mrSegNet, epoch,ePOCH, save_DIR):

    total_dice = np.zeros((3,))
    total_Iou = np.zeros((3,))
    total_avghausdorff = np.zeros((3,))
    total_itkdice = np.zeros((3,))
    test_dice_dict = {}
    test_assd_dict = {}
    mrSegNet.eval()
    for imagename in testfile_list:
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

        data = torch.from_numpy(np.expand_dims(np.expand_dims(npimg,0),0))
        label = torch.from_numpy(np.expand_dims(nplabs,0))

        data = data.cuda()
        label = label.cuda()
        output,_,_, _, _,_,_, _,_,_,_, _, _,_,_, _,_ = mrSegNet(data,0)

        truemax, truearg = torch.max(output, 1, keepdim=False)
        truearg = truearg.detach().cpu().numpy()
        # if epoch==ePOCH-1:
        #
        #     truelabel = (truearg == 1) * 205  + (truearg == 2) * 500 # \
        #                # (truearg == 4) * 250 + (truearg == 5) * 850 + (truearg == 6) * 820 + (truearg == 7) * 0
        #     scipy.misc.imsave('%s/mr_%s_testout.jpg'%(save_IMG_DIR,filename[0]), np.squeeze(truelabel))
        dice = dice_compute(truearg,label.cpu().numpy())
        Iou = IOU_compute(truearg,label.cpu().numpy())
        overlap_result, surface_distance_result = Hausdorff_compute(truearg,label.cpu().numpy(),itklab.GetSpacing()[0:2])

        total_dice = np.vstack((total_dice,dice))
        total_Iou = np.vstack((total_Iou,Iou))
        total_avghausdorff = np.vstack((total_avghausdorff,surface_distance_result))
        total_itkdice = np.vstack((total_itkdice,overlap_result))

        test_dice_dict[labelname.split('/')[-1]] = dice
        test_assd_dict[labelname.split('/')[-1]] = surface_distance_result

    meanDice = np.mean(total_dice[1:],axis=0)
    stdDice = np.std(total_dice[1:],axis=0)

    criterion_dice = np.mean(meanDice[1:])

    meanIou = np.mean(total_Iou[1:],axis=0)
    stdIou = np.std(total_Iou[1:],axis=0)

    mean_avghausdorff = np.mean(total_avghausdorff[1:], axis=0)
    std_avghausdorff = np.std(total_avghausdorff[1:], axis=0)

    mean_itkdice = np.mean(total_itkdice[1:], axis=0)
    std_itkdice = np.std(total_itkdice[1:], axis=0)

    with open("%s/mr_testout_index.txt" % save_DIR, "a") as f:
        f.writelines(["\nepoch:", str(epoch), " ", "","meanDice:",""\
                         ,str(meanDice.tolist()),"stdDice:","",str(stdDice.tolist()),"","meanIou:","",str(meanIou.tolist()),"stdIou:","",str(stdIou.tolist()), \
                          "", "meanavghausdorff:", "", str(mean_avghausdorff.tolist()), "stdavghausdorff:", "", str(std_avghausdorff.tolist()), \
                          "", "meanitkdice:", "", str(mean_itkdice.tolist()), "stditkdice:", "", str(std_itkdice.tolist())])

    return criterion_dice,test_dice_dict,test_assd_dict,total_dice[1:],total_avghausdorff[1:]




def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main(configs):



    SAVE_DIR = configs.prefix + '/save_train_param' + str(int(round(math.log(configs.KLDLamda, 10)))) + str(
        int(round(math.log(configs.BCELamda, 10))))

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)



    criterion = 0
    best_epoch = 0
    dice_dict = None
    assd_dict =None
    dice_ny = None
    assd_ny = None


    vaeencoder = VAE()
    vaeencoder = vaeencoder.cuda()



    target_vaedecoder = Decoder(16)
    target_vaedecoder = target_vaedecoder.cuda()

    target_down2_vaedecoder = Decoder(32)
    target_down2_vaedecoder = target_down2_vaedecoder.cuda()

    target_down4_vaedecoder = Decoder(64)
    target_down4_vaedecoder = target_down4_vaedecoder.cuda()


    DistanceNet = Feature_Distribution_Distance_func(configs.KERNEL,configs.KERNEL)  #64,Num_Feature2,(12,12)
    DistanceNet = DistanceNet.cuda()
    #DistanceNet2 = nn.DataParallel(DistanceNet2, device_ids=[0,1])


    DA_optim = torch.optim.Adam([{'params': vaeencoder.parameters()},{'params': target_vaedecoder.parameters()},{'params': target_down2_vaedecoder.parameters()},{'params': target_down4_vaedecoder.parameters()}],lr=configs.LR,weight_decay=configs.WEIGHT_DECAY)


    SourceData = DataSet_Train(configs.Source_Dir)
    SourceData_loader = DataLoader(SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)

    TargetData = DataSet_Train(configs.Target.Dir)
    TargetData_loader = DataLoader(TargetData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)



    SAVE_DIR_CV=SAVE_DIR
    if not os.path.exists(SAVE_DIR_CV):
        os.mkdir(SAVE_DIR_CV)


    vaeencoder.apply(init_weights)
    target_vaedecoder.apply(init_weights)
    target_down2_vaedecoder.apply(init_weights)
    target_down4_vaedecoder.apply(init_weights)


    for epoch in range(configs.EPOCHs):
        vaeencoder.train()
        target_vaedecoder.train()
        target_down2_vaedecoder.train()
        target_down4_vaedecoder.train()
        ADA_Train( SourceData_loader,TargetData_loader,vaeencoder,target_vaedecoder,target_down2_vaedecoder,target_down4_vaedecoder,1.0,DistanceNet,configs.LR,configs.KLDLamda,configs.PredLamda,configs.BCELamda,configs.Dislamda,epoch,DA_optim, SAVE_DIR_CV)
        vaeencoder.eval()
        criterion_dice, test_dice_dict_temp, test_assd_dict_temp, dice_numpy, assd_numpy = SegNet_test_mr(configs.Target_Dir_Vali, vaeencoder, epoch, configs.EPOCHs, SAVE_DIR_CV)

        if criterion_dice > criterion:
            best_epoch = epoch
            criterion = criterion_dice
            dice_dict = test_dice_dict_temp
            assd_dict = test_assd_dict_temp
            dice_ny = dice_numpy
            assd_ny = assd_numpy
            torch.save(vaeencoder.state_dict(), os.path.join(SAVE_DIR_CV, 'encoder_param.pkl'))
            torch.save(target_vaedecoder.state_dict(), os.path.join(SAVE_DIR_CV, 'decoderB_param.pkl'))
            torch.save(target_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR_CV, 'decoderBdown2_param.pkl'))
            torch.save(target_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR_CV, 'decoderBdown4_param.pkl'))



    print ('\n')
    print ('best epoch:%d' % (best_epoch))
    with open("%s/mr_testout_index.txt" % (SAVE_DIR_CV), "a") as f:
        f.writelines(["\n\nbest epoch:%d" % (best_epoch)])
    np.save(SAVE_DIR + '/test_dice_dict_best.npy', np.array(dice_dict))
    np.save(SAVE_DIR + '/test_assd_dict_best.npy', np.array(assd_dict))

    mean_avghausdorff = np.mean(assd_ny[1:], axis=0)
    std_avghausdorff = np.std(dice_ny[1:], axis=0)


    with open("%s/mr_testout_index.txt" % SAVE_DIR, "a") as f:
        f.writelines(["meanavghausdorff:", "", str(mean_avghausdorff.tolist()), "stdavghausdorff:", "",
                      str(std_avghausdorff.tolist())])



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    parser.add_argument('--LR', type=float, default=0.01)
    parser.add_argument('--PredLamda', type=float, default=1e3)
    parser.add_argument('--Dislamda', type=float, default=1e1)
    parser.add_argument('--KLDLamda', type=float, default=1e0)
    parser.add_argument('--BCELamda', type=float, default=1e0)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-5)

    parser.add_argument('--EPOCHS', type=int, default=50)
    parser.add_argument('--WORKERSNUM', type=int, default=20)
    parser.add_argument('--BatchSize', type=int, default=10)
    parser.add_argument('--KERNEL', type=int, default=1)

    parser.add_argument('--Source_Dir', type=str, default='./Dataset/CT/Train')
    parser.add_argument('--Target_Dir_Train', type=str, default='./Dataset/MR/Train')
    parser.add_argument('--Target_Dir_Vali', type=str, default='./Dataset/MR/Vali')
    parser.add_argument('--prefix', type=str, default='./')


    CONFIGs = parser.parse_args()

    main(CONFIGs)
