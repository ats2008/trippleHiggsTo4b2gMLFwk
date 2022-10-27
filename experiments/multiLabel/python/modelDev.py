#!/usr/bin/env python
# coding: utf-8

import argparse,copy

import uproot3 as upr

import pandas as pd

import torch
import torch.utils.data as data
import sys,os



import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import random
import math
import json
from functools import partial

import MLUtils as mlUtil
from MLUtils import printHead

from model import trippleHDataset
from model import trippleHNonResonatModel


# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


def getDataset(dataFileNames,inputVarList,labelVars,isvalidMaskVar=None,
               testTrainSplit=0.1,randomize=True,
               seedVal=42,nSigEvts=-1,nBkgEvts=-1):
    bkgDataset,bkgLabl,bkgMask=mlUtil.readDataset(dataFileNames['bkg'],labelVars,
                                          'bkg_13TeV_TrippleHTag_0_background',
                                          inputVarList,
                                          isvalidMaskVar=isvalidMaskVar,
                                          NEVTS=nBkgEvts)
    sigDataset,sigLabl,sigMask=mlUtil.readDataset(dataFileNames['sig'],labelVars,
                                          'ggHHH_125_13TeV_allRecoed',
                                          inputVarList,
                                          isvalidMaskVar=isvalidMaskVar,
                                          NEVTS=nSigEvts)

    dataset=torch.cat([bkgDataset,sigDataset],0)
    label=torch.cat([bkgLabl,sigLabl],0)
    mask=torch.cat([bkgMask,sigMask],0)

    bkgMem=sys.getsizeof(bkgDataset.storage())/1e6
    sigMem=sys.getsizeof(sigDataset.storage())/1e6
    print("Total number of Signal Events     : ",sigDataset.shape[0]," [ ",sigMem," MB ]")
    print("Total number of Background Events : ",bkgDataset.shape[0]," [ ",bkgMem," MB]")
    allEvtCount=sigDataset.shape[0]+bkgDataset.shape[0]
    if randomize:
        print("   randomizing the dataset ! (with numpy seed : ",seedVal,")")
        np.random.seed(seedVal)
        permut=np.random.permutation(dataset.shape[0])
        dataset=dataset[permut]
        label=label[permut]
        mask=mask[permut]
    test_count = int(min(bkgDataset.shape[0],sigDataset.shape[0])/10.0)
    data_test   = dataset[:test_count]
    label_test = label[:test_count]
    mask_test = mask[:test_count]
    
    dataset =dataset[test_count:]
    label=label[test_count:]
    mask=mask[test_count:]
    print("train dataset shape ",dataset.shape)
    print("test dataset shape ",data_test.shape)
    print("ASSERTING NO EVT LOSS  ",data_test.shape[0]+dataset.shape[0], "== ",allEvtCount)
    return {'data':dataset,'labels':label,'mask':mask,
            'data_test':data_test,'label_test':label_test,'mask_test':mask_test}


def getTestTrainDataLoaders(dataFileNames,inputVarList,labelVars,isvalidMaskVar=None,
                            nSigEvts=-1,nBkgEvts=-1,getDataLoader=True):
    text="Loading Data"
    
    dataset=getDataset(dataFileNames,inputVarList,labelVars,
                       isvalidMaskVar=isvalidMaskVar,
                       nSigEvts=nSigEvts,nBkgEvts=nBkgEvts,
                       testTrainSplit=0.1,randomize=True)
    
    # ### Test Train/Validation Split
    
    text="Making Test train Split"
    printHead(text)
    
    data_train   = dataset['data']
    labels_train = dataset['labels']
    masks_train = dataset['mask']
    
    print("data shape : ",data_train.shape)

    print("labels shape : ",labels_train.shape)
    print("Number of uniquie classes in training set : ",np.unique(labels_train))
    signalMask=( torch.sum(labels_train,dim=1) > 0  ).squeeze()
    bkgMask   =( torch.sum(labels_train,dim=1) < 0.5).squeeze()
    sorted_indices={}
    sorted_indices[1]=np.arange(labels_train.shape[0])[signalMask]
    sorted_indices[0]=np.arange(labels_train.shape[0])[bkgMask]
    
    
    print("Total number of events for training : ",labels_train.shape[0],"(",sum(signalMask),"+",sum(bkgMask),")")
    
    num_val_exmps = int(min(len(sorted_indices[0]),len(sorted_indices[1]))/10)
    print("Setting number of Vaidation sig[bkg] elements as  ",num_val_exmps)
    
    # Get image indices for validation and training
    val_indices   = sorted_indices[0][:num_val_exmps]
    val_indices   = np.concatenate([val_indices,sorted_indices[1][:num_val_exmps]])
    
    train_indices = sorted_indices[0][num_val_exmps:]
    train_indices   = np.concatenate([train_indices,sorted_indices[1][num_val_exmps:]])
    
    # Group corresponding image features and labels
    train_feats, train_labels, train_masks = data_train[train_indices], labels_train[train_indices], masks_train[train_indices]
    val_feats,   val_labels , val_masks  = data_train[val_indices],   labels_train[val_indices], masks_train[val_indices]
    
    print("train_feats.shape : ",train_feats.shape)
    print("val_feats.shape   : ",val_feats.shape)
    
    print("ASSERT    : ",val_feats.shape[0]+train_feats.shape[0]," == ",labels_train.shape[0])
    
    test_feats = dataset['data_test']
    test_labels = dataset['label_test']
    test_masks = dataset['mask_test']

    if not getDataLoader:
        allDataDict={'data_train' : train_feats,'train_labels':train_labels,
                'data_val' : val_feats,'val_labels':val_labels,
                'data_test' : test_feats,'test_labels':test_labels
               }
        return allDataDict
    
    train_dataset = trippleHDataset(train_feats, train_labels,train_masks, train=True)
    val_dataset   = trippleHDataset(val_feats  ,   val_labels,  val_masks, train=False)
    test_dataset  = trippleHDataset(test_feats ,  test_labels, test_masks, train=False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
    val_loader   = data.DataLoader(val_dataset,   batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader  = data.DataLoader(test_dataset,  batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    return train_loader,val_loader,test_loader




def train_hhhVsQCD(train_loader,val_loader,test_loader,
                   maxEpoch,version=None,
                   CHECKPOINT_PATH='./chkpt.tmp',
                   **kwargs):
    
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "trippleHiggsVsQCD")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=maxEpoch,
                         gradient_clip_val=2)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    trainer.val_check_interval=0.05
    hpars=copy.deepcopy(kwargs)
    if 'inputVarList' in hpars:
        hpars['inputVarList']=str(hpars['inputVarList'])
    trainer.logger.experiment.add_hparams(hpars,metric_dict={})

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "trippleHiggsVsQCD.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = trippleHNonResonatModel.load_from_checkpoint(pretrained_filename)
    else:
        model = trippleHNonResonatModel(max_iters=trainer.max_epochs*len(train_loader), **kwargs)

        printHead("Traing the model for "+str(maxEpoch) +' epochs')
        trainer.fit(model, train_loader, val_loader)
        model = trippleHNonResonatModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result   = trainer.test(model, val_loader, verbose=False)
    test_result  = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"], "train_acc": train_result[0]["test_acc"]}
    model = model.to(device)
    return model, trainer,result

def load_model(CHECKPOINT_PATH='./chkpt.tmp',**kwargs):
    
    # Create a PyTorch Lightning trainer with the generation callback
    if not os.path.exists(CHECKPOINT_PATH):
        raise Exception(CHECKPOINT_PATH+" do not exists ! ")
    
    if os.path.isfile(CHECKPOINT_PATH):
        print("Found pretrained model, loading...")
        model = trippleHNonResonatModel.load_from_checkpoint(CHECKPOINT_PATH)
        model.eval()
    else:
        raise Exception(CHECKPOINT_PATH+" is not a file ! ")
        
    return model

if __name__=="__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Tag",default='tmp')
    parser.add_argument("--remark", help="Remark",default='Nothing specific ! LOL')
    parser.add_argument("--chk", help="Checkpoint",default='workarea/')
    args = parser.parse_args()
    
    CHECKPOINT_PATH=(args.chk + '/' +args.tag).replace('//','/')
    print("CHECKPOINT_PATH : ",CHECKPOINT_PATH)
    tag=args.tag

    inputVarList=[ ]
    for i in range(8):
        inputVarList.append('jet_'+str( i )+'_pt')
        inputVarList.append('jet_'+str( i )+'_eta')
        inputVarList.append('jet_'+str( i )+'_phi')
        inputVarList.append('jet_'+str( i )+'_mass')
        inputVarList.append('jet_'+str( i )+'_deepCSVScore')
        inputVarList.append('jet_'+str( i )+'_bJetRegCorr')
        inputVarList.append('jet_'+str( i )+'_bJetRegRes')
        inputVarList.append('jet_'+str( i )+'_drWithDP')
        inputVarList.append('jet_'+str( i )+'_drWithDP_leadG')
        inputVarList.append('jet_'+str( i )+'_dr_WithDPSubleadG')
        inputVarList.append('jet_'+str( i )+'_dEta_WithDP')
        inputVarList.append('jet_'+str( i )+'_dEta_WithDPLeadG')
        inputVarList.append('jet_'+str( i )+'_dEta_WithDPSubleadG')
        inputVarList.append('jet_'+str( i )+'_dPhi_WithDP')
        inputVarList.append('jet_'+str( i )+'_dPhi_WithDPLeadG')
        inputVarList.append('jet_'+str( i )+'_dPhi_WithDPSubleadG')
        inputVarList.append('jet_'+str( i )+'_mass_WithDP')
        inputVarList.append('jet_'+str( i )+'_mass_WithDPLeadG')
        inputVarList.append('jet_'+str( i )+'_mass_WithDPSubleadG')
    
    
    labelVars=['label_idx_'+str(i) for i in range(8)]
    isvalidMaskVar=['jet_'+str(i)+'_isValid' for i in range(8)]
    
    print("Number of Input variables : ",len(inputVarList))
    print("Number of Input Labels : ",len(labelVars))
    
    dataFileNames={
        'sig':{
            'ggHHH' : '/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018//ggHHH/ml_ggHHH_1p0.root'
        },
        'bkg':{
            'ggM80Jbox2bjet' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton2B_1p0.root',
            'ggM80Jbox1bjet' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton1B_1p0.root',
            'ggM80Inc' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton_1p0.root'
        }
    }

    train_loader,val_loader,test_loader = getTestTrainDataLoaders(dataFileNames,inputVarList,
                                                                  labelVars,isvalidMaskVar=isvalidMaskVar,
                                                                  nSigEvts=-1,nBkgEvts=-1)

    train_sample,mask,_=train_loader.dataset[0];

    print("Tarining The model !!  ",train_sample[-1].shape)

    hhhVsQCD_model,model_trainer, hhhVsQCD_result = train_hhhVsQCD( train_loader , val_loader,test_loader,
                                                  maxEpoch=200,
                                                  inputVarList=inputVarList,
                                                  remark=tag + args.remark,
                                                  CHECKPOINT_PATH=CHECKPOINT_PATH,
	    				                          input_dim=19,
                                                  model_dim=128*8,
                                                  num_heads=32,
                                                  num_classes=5,
                                                  num_layers=16,
                                                  dropout=0.1,
                                                  input_dropout=0.0,
                                                  lr=5e-4,
                                                  warmup=200
                                              )
    print("Training Accuracy    : ",hhhVsQCD_result['train_acc'])
    print("Validation  Accuracy : ",hhhVsQCD_result['val_acc'])
    print("Testing  Accuracy    : ",hhhVsQCD_result['test_acc'])

