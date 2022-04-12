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
TestDir=['./Patch192/LGE_Test/','./Patch192/LGE_Vali/']
BatchSize = 10
KERNEL=20

prefix='.'


def ADA_Train(Train_LoaderA,Train_LoaderB,encoder,decoderA,decoderAdown2,decoderAdown4,decoderB,decoderBdown2,decoderBdown4,gate,DistanceNet,lr,kldlamda,predlamda,dislamda,epoch,optim, savedir):
    lr=lr*(0.9**(epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr


    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)

    i=0


    while i<len(A_iter) and i<len(B_iter):
        ct,ct_down2,ct_down4,label,label_down2,label_down4 ,info_ct= A_iter.next()
        mr,mr_down2,mr_down4,info_mr= B_iter.next()

        ct= ct.cuda()
        ct_down2= ct_down2.cuda()
        ct_down4= ct_down4.cuda()
        #info_ct = info_ct.cuda()

        mr= mr.cuda()
        mr_down4= mr_down4.cuda()
        mr_down2= mr_down2.cuda()
        #info_mr = info_mr.cuda()

        label= label.cuda()
        label_onehot =torch.FloatTensor(label.size(0), 4,label.size(1),label.size(2)).cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1, label.unsqueeze(dim=1), 1)

        label_down2= label_down2.cuda()
        label_down2_onehot =torch.FloatTensor(label_down2.size(0), 4,label_down2.size(1),label_down2.size(2)).cuda()
        label_down2_onehot.zero_()
        label_down2_onehot.scatter_(1, label_down2.unsqueeze(dim=1), 1)

        label_down4= label_down4.cuda()
        label_down4_onehot =torch.FloatTensor(label_down4.size(0), 4,label_down4.size(1),label_down4.size(2)).cuda()
        label_down4_onehot.zero_()
        label_down4_onehot.scatter_(1, label_down4.unsqueeze(dim=1), 1)

        fusionseg,_, out_ct,feat_ct, mu_ct,logvar_ct, _, outdown2_ct,featdown2_ct, mudown2_ct,logvardown2_ct,_, outdown4_ct,featdown4_ct, mudown4_ct,logvardown4_ct,info_pred_ct= encoder(ct,gate)
        #info_pred_ct = Infonet(info_pred_ct)

        info_cri = nn.CrossEntropyLoss().cuda()
        #infoloss_ct = info_cri(info_pred_ct,info_ct)

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

        recon_ct=decoderA(feat_ct)
        BCE_ct = F.mse_loss(recon_ct, ct) #binary_cross_entropy
        KLD_ct = -0.5 * torch.mean(1 + logvar_ct - mu_ct.pow(2) - logvar_ct.exp())

        recondown2_ct=decoderAdown2(featdown2_ct)
        BCE_down2_ct = F.mse_loss(recondown2_ct, ct_down2)
        KLD_down2_ct = -0.5 * torch.mean(1 + logvardown2_ct - mudown2_ct.pow(2) - logvardown2_ct.exp())

        recondown4_ct=decoderAdown4(featdown4_ct)
        BCE_down4_ct = F.mse_loss(recondown4_ct, ct_down4)
        KLD_down4_ct = -0.5 * torch.mean(1 + logvardown4_ct - mudown4_ct.pow(2) - logvardown4_ct.exp())

        _,pred_mr, _,feat_mr, mu_mr,logvar_mr, preddown2_mr, _,featdown2_mr, mudown2_mr,logvardown2_mr,preddown4_mr, _,featdown4_mr, mudown4_mr,logvardown4_mr,info_pred_mr= encoder(mr,gate)
        #info_pred_mr = Infonet(info_pred_mr)

        #infoloss_mr = info_cri(info_pred_mr,info_mr)

        recon_mr=decoderB(feat_mr)
        BCE_mr = F.mse_loss(recon_mr, mr)
        KLD_mr = -0.5 * torch.mean(1 + logvar_mr - mu_mr.pow(2) - logvar_mr.exp())

        recondown2_mr=decoderBdown2(featdown2_mr)
        BCE_down2_mr = F.mse_loss(recondown2_mr, mr_down2)
        KLD_down2_mr = -0.5 * torch.mean(1 + logvardown2_mr - mudown2_mr.pow(2) - logvardown2_mr.exp())

        recondown4_mr=decoderBdown4(featdown4_mr)
        BCE_down4_mr = F.mse_loss(recondown4_mr, mr_down4)
        KLD_down4_mr = -0.5 * torch.mean(1 + logvardown4_mr - mudown4_mr.pow(2) - logvardown4_mr.exp())

        distance_loss = DistanceNet(feat_ct,feat_mr)
        #distance_down2_loss = DistanceNet(featdown2_ct,featdown2_mr)
        #distance_down4_loss = DistanceNet(featdown4_ct,featdown4_mr)

        meanloss1 = torch.mean((feat_ct.mean(dim=0, keepdim=True) - feat_mr.mean(dim=0, keepdim=True)) ** 2)
        meanloss2 = torch.mean((featdown2_ct.mean(dim=0, keepdim=True) - featdown2_mr.mean(dim=0, keepdim=True)) ** 2)
        meanloss3 = torch.mean((featdown4_ct.mean(dim=0, keepdim=True) - featdown4_mr.mean(dim=0, keepdim=True)) ** 2)



        balanced_loss = BCE_mr+torch.mul(KLD_mr,kldlamda)+BCE_ct+torch.mul(KLD_ct,kldlamda)+torch.mul(distance_loss,dislamda)+predlamda*(segloss_output+fusionsegloss_output)+ \
                        BCE_down2_ct + torch.mul(KLD_down2_ct, kldlamda) + BCE_down2_mr + torch.mul(KLD_down2_mr, kldlamda) + predlamda * segdown2loss_output+ \
                        BCE_down4_ct + torch.mul(KLD_down4_ct, kldlamda) + BCE_down4_mr + torch.mul(KLD_down4_mr, kldlamda) + predlamda * segdown4loss_output+1e3*(meanloss1+meanloss2+meanloss3)

        optim.zero_grad()
        balanced_loss.backward()
        optim.step()

        # f = open(save_loss_txt, 'a')
        # f.write('{0:.4f}\n'.format(balanced_loss))
        # f.close()
        #
        # f = open(save_cfd_txt, 'a')
        # f.write('{0:.4f}\n'.format(distance_loss))
        # f.close()

        if i % 20 == 0:
            print('epoch %d , %d th iter; seglr,ADA_totalloss,segloss,distance_loss1,distance_loss2: %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'\
                  % (epoch, i,lr, balanced_loss.item(),BCE_mr.item(),KLD_mr.item(),BCE_ct.item(),KLD_ct.item(),fusionsegloss_output.item(),segloss_output.item(),segdown2loss_output.item(),segdown4loss_output.item(),distance_loss.item()))

        i=i+1



def SegNet_test_mr(test_dir, mrSegNet, gate,epoch, save_DIR):
    vali_dice = 0
    vali_assd = 0
    test_dice = 0
    test_assd = 0
    test_dice_dict = {}
    test_assd_dict = {}
    for dir in test_dir:
        labsname = glob.glob(dir + '*manual.nii*')

        total_dice = np.zeros((4,))
        total_Iou = np.zeros((4,))

        total_overlap =np.zeros((1,4, 5))
        total_surface_distance=np.zeros((1,4, 5))

        num = 0
        mrSegNet.eval()
        dice_dict={}
        assd_dict = {}
        for i in range(len(labsname)):
            itklab = sitk.ReadImage(labsname[i])
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            imgname = labsname[i].replace('_manual.nii', '.nii')
            itkimg = sitk.ReadImage(imgname)
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)

            data=torch.from_numpy(np.expand_dims(npimg,axis=1)).type(dtype=torch.FloatTensor).cuda()

            truearg = np.zeros((data.size(0),data.size(2),data.size(3)))

            for slice in range(data.size(0)):
                output, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = mrSegNet(data[slice:slice + 1, :, :, :], gate)

                truemax, truearg0 = torch.max(output, 1, keepdim=False)
                truearg[slice:slice+1,:,:] = truearg0.detach().cpu().numpy()

            dice = dice_compute(truearg,nplab)
            Iou = IOU_compute(truearg,nplab)
            overlap_result, surface_distance_result = Hausdorff_compute(truearg,nplab,itkimg.GetSpacing())

            total_dice = np.vstack((total_dice,dice))
            total_Iou = np.vstack((total_Iou,Iou))

            total_overlap = np.concatenate((total_overlap,overlap_result),axis=0)
            total_surface_distance = np.concatenate((total_surface_distance,surface_distance_result),axis=0)

            num+=1

            dice_dict[labsname[i].split('/')[-1]]= dice
            assd_dict[labsname[i].split('/')[-1]] = surface_distance_result[0,:,1]

        if num==0:
            return
        else:
            meanDice = np.mean(total_dice[1:],axis=0)
            stdDice = np.std(total_dice[1:],axis=0)

            meanIou = np.mean(total_Iou[1:],axis=0)
            stdIou = np.std(total_Iou[1:],axis=0)

            mean_overlap = np.mean(total_overlap[1:], axis=0)
            std_overlap = np.std(total_overlap[1:], axis=0)

            mean_surface_distance = np.mean(total_surface_distance[1:], axis=0)
            std_surface_distance = np.std(total_surface_distance[1:], axis=0)

            if 'Vali' in dir:
                phase='validate'
                vali_dice = np.mean(meanDice[1:])
                vali_assd = np.mean(mean_surface_distance[1:,1])
            else:
                phase='test'
                test_dice_dict = dice_dict
                test_assd_dict = assd_dict
                test_dice = np.mean(meanDice[1:])
                test_assd = np.mean(mean_surface_distance[1:, 1])
            with open("%s/lge_testout_index.txt" % (save_DIR), "a") as f:
                f.writelines(["\n\nepoch:", str(epoch), " ",phase," ", "\n","meanDice:",""\
                                 ,str(meanDice.tolist()),"stdDice:","",str(stdDice.tolist()),"","\n","meanIou:","",str(meanIou.tolist()),"stdIou:","",str(stdIou.tolist()), \
                                  "", "\n\n","jaccard, dice, volume_similarity, false_negative, false_positive:", "\n","mean:", str(mean_overlap.tolist()),"\n", "std:", "", str(std_overlap.tolist()), \
                                  "", "\n\n","hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance:", "\n","mean:", str(mean_surface_distance.tolist()), "\n","std:", str(std_surface_distance.tolist())])
    return vali_dice,vali_assd,test_dice,test_assd,test_dice_dict,test_assd_dict


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="2"

    cudnn.benchmark = True
    vaeencoder = VAE()
    vaeencoder = vaeencoder.cuda()

    source_vaedecoder = Decoder(16)
    source_vaedecoder = source_vaedecoder.cuda()

    source_down2_vaedecoder = Decoder(32)
    source_down2_vaedecoder = source_down2_vaedecoder.cuda()

    source_down4_vaedecoder = Decoder(64)
    source_down4_vaedecoder = source_down4_vaedecoder.cuda()

    target_vaedecoder = Decoder(16)
    target_vaedecoder = target_vaedecoder.cuda()

    target_down2_vaedecoder = Decoder(32)
    target_down2_vaedecoder = target_down2_vaedecoder.cuda()

    target_down4_vaedecoder = Decoder(64)
    target_down4_vaedecoder = target_down4_vaedecoder.cuda()

    #Infonet = InfoNet().cuda()

    DistanceNet = Feature_Distribution_Distance_func(KERNEL,KERNEL)  #64,Num_Feature2,(12,12)
    DistanceNet = DistanceNet.cuda()
    #DistanceNet2 = nn.DataParallel(DistanceNet2, device_ids=[0,1])


    DA_optim = torch.optim.Adam([{'params': vaeencoder.parameters()},{'params': source_vaedecoder.parameters()},{'params': source_down2_vaedecoder.parameters()},{'params': source_down4_vaedecoder.parameters()},{'params': target_vaedecoder.parameters()},{'params': target_down2_vaedecoder.parameters()},{'params': target_down4_vaedecoder.parameters()}],lr=LR,weight_decay=WEIGHT_DECAY)

    SourceData = C0_TrainSet()
    SourceData_loader = DataLoader(SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True,drop_last = True)

    TargetData = LGE_TrainSet()
    TargetData_loader = DataLoader(TargetData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True,drop_last = True)


    # TestData = LabeledDataSet(modality='mr',stage='test')
    # TestData_loader = DataLoader(TestData, batch_size=1, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)
    PredLamda=1e3
    DisLamdaList=[1e-5]


    for DisLamda in DisLamdaList:


        print ('PredLamda'+str(PredLamda)+',  DisLamda'+str(DisLamda))
        print ('\n')
        print ('\n')
        SAVE_DIR=prefix+'/bs'+str(BatchSize)+'_DisLamda'+str(int(round(math.log(DisLamda,10))))
        SAVE_IMG_DIR=prefix+'/label_bs'+str(BatchSize)+'_DisLamda'+str(int(round(math.log(DisLamda,10))))
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not os.path.exists(SAVE_IMG_DIR):
            os.mkdir(SAVE_IMG_DIR)
        vaeencoder.apply(init_weights)
        source_vaedecoder.apply(init_weights)
        source_down2_vaedecoder.apply(init_weights)
        source_down4_vaedecoder.apply(init_weights)
        target_vaedecoder.apply(init_weights)
        target_down2_vaedecoder.apply(init_weights)
        target_down4_vaedecoder.apply(init_weights)

        criterion=0
        best_epoch=0
        # save_loss_txt = SAVE_DIR + '/loss_iter.txt'
        # save_cfd_txt = SAVE_DIR + '/cfdloss_iter.txt'
        # save_vali_dice_txt = SAVE_DIR + '/vali_dice_epoch.txt'
        # save_vali_assd_txt = SAVE_DIR + '/vali_assd_epoch.txt'
        # save_test_dice_txt = SAVE_DIR + '/test_dice_epoch.txt'
        # save_test_assd_txt = SAVE_DIR + '/test_assd_epoch.txt'
        for epoch in range(EPOCH):
            vaeencoder.train()
            source_vaedecoder.train()
            source_down2_vaedecoder.train()
            source_down4_vaedecoder.train()
            target_vaedecoder.train()
            target_down2_vaedecoder.train()
            target_down4_vaedecoder.train()
            ADA_Train(SourceData_loader,TargetData_loader,vaeencoder,source_vaedecoder,source_down2_vaedecoder,source_down4_vaedecoder,target_vaedecoder,target_down2_vaedecoder,target_down4_vaedecoder,1.0,DistanceNet,LR,KLDLamda,PredLamda,DisLamda,epoch,DA_optim, SAVE_DIR)
            vaeencoder.eval()
            vali_dice, vali_assd, test_dice, test_assd, test_dice_dict, test_assd_dict =SegNet_test_mr(TestDir, vaeencoder,0, epoch, SAVE_DIR)
            # f = open(save_vali_dice_txt, 'a')
            # f.write('{0:.4f}\n'.format(vali_dice))
            # f.close()
            #
            # f = open(save_vali_assd_txt, 'a')
            # f.write('{0:.4f}\n'.format(vali_assd))
            # f.close()
            #
            # f = open(save_test_dice_txt, 'a')
            # f.write('{0:.4f}\n'.format(test_dice))
            # f.close()
            #
            # f = open(save_test_assd_txt, 'a')
            # f.write('{0:.4f}\n'.format(test_assd))
            # f.close()
            if vali_dice > criterion:
                best_epoch = epoch
                criterion = vali_dice
                torch.save(vaeencoder.state_dict(), os.path.join(SAVE_DIR, 'encoder_param.pkl'))
                torch.save(source_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderA_param.pkl'))
                torch.save(source_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderAdown2_param.pkl'))
                torch.save(source_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderAdown4_param.pkl'))
                torch.save(target_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderB_param.pkl'))
                torch.save(target_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderBdown2_param.pkl'))
                torch.save(target_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderBdown4_param.pkl'))
                np.save(SAVE_DIR + '/test_dice_dict_best.npy', np.array(test_dice_dict))
                np.save(SAVE_DIR + '/test_assd_dict_best.npy', np.array(test_assd_dict))
        print ('\n')
        print ('\n')
        print ('best epoch:%d' % (best_epoch))
        with open("%s/lge_testout_index.txt" % (SAVE_DIR), "a") as f:
            f.writelines(["\n\nbest epoch:%d" % (best_epoch)])


if __name__ == '__main__':
    main()
