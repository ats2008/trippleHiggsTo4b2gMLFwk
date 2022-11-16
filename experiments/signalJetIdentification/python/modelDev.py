#!/usr/bin/env python
# coding: utf-8

import argparse,copy

import uproot3 as upr

import pandas as pd

import torch
import torch.utils.data as data
import sys,os
import uuid
from datetime import datetime 


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


import numpy as np
import random
import math
import json
import pickle,uuid
from functools import partial

import MLUtils as mlUtil
from MLUtils import printHead

import modelEval
import variablePlotter
from model import trippleHDataset
from model import trippleHNonResonatModel


# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import NeptuneLogger
neptuneApiKey=''
neptuneProject=''
base=''
if 'NEPTUNE_API_TOKEN' in os.environ:
    neptuneApiKey=os.environ['NEPTUNE_API_TOKEN']
if 'NEPTUNE_PROJECT' in os.environ:
    neptuneProject=os.environ['NEPTUNE_PROJECT']
if 'BASE_DIR' in os.environ:
    base=os.environ['BASE_DIR']
# Neptune

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
def getDataset(dataFileNames,inputVarList,labelVars,
               isvalidMaskVar=None,otherVars={},
               testTrainSplit=0.1,randomize=True,
               seedVal=42,nSigEvts=-1,nBkgEvts=-1):

    dset=mlUtil.readDataset(dataFileNames['bkg'],labelVars,
                                          'bkg_13TeV_TrippleHTag_0_background',
                                          inputVarList,otherVars=otherVars,
                                          isvalidMaskVar=isvalidMaskVar,
                                          NEVTS=nBkgEvts)
    bkgDataset  = dset['data']
    bkgLabl     = dset['label']
    bkgMask     = dset['mask']
    bkgExtraVars= dset['otherVars']
    
    dset=mlUtil.readDataset(dataFileNames['sig'],labelVars,
                                          'ggHHH_125_13TeV_allEvents',
#                                          'ggHHH_125_13TeV_allRecoed',
                                          inputVarList,otherVars=otherVars,
                                          isvalidMaskVar=isvalidMaskVar,
                                          NEVTS=nSigEvts)
    sigDataset  = dset['data']
    sigLabl     = dset['label']
    sigMask     = dset['mask']
    sigExtraVars= dset['otherVars']

    dataset=torch.cat([bkgDataset,sigDataset],0)
    label=torch.cat([bkgLabl,sigLabl],0)
    mask=torch.cat([bkgMask,sigMask],0)
    extraVars={}
    for tag in bkgExtraVars:
        if tag not in sigExtraVars:
            raise Exception("tag : ",tag, " in bkg set but not in sig set")
            continue
        extraVars[tag]=torch.cat([bkgExtraVars[tag],sigExtraVars[tag]],0)
    
    
    bkgMem=sys.getsizeof(bkgDataset.storage())/1e6
    sigMem=sys.getsizeof(sigDataset.storage())/1e6
    extraVarMem=sum([sys.getsizeof(extraVars[tag].storage()) for tag in extraVars])/1e6
    print("Total number of Signal Events     : ",sigDataset.shape[0]," [ ",sigMem," MB ]")
    print("Total number of Background Events : ",bkgDataset.shape[0]," [ ",bkgMem," MB]")
    print("Total memory occupied by extra var : ",extraVarMem," MB]")
    allEvtCount=sigDataset.shape[0]+bkgDataset.shape[0]
    if randomize:
        print("   randomizing the dataset ! (with numpy seed : ",seedVal,")")
        np.random.seed(seedVal)
        permut=np.random.permutation(dataset.shape[0])
        dataset=dataset[permut]
        label=label[permut]
        mask=mask[permut]
        for tag in extraVars:
            extraVars[tag]=extraVars[tag][permut]
    test_count = int(min(bkgDataset.shape[0],sigDataset.shape[0])*testTrainSplit)
    
    data_test   = dataset[:test_count]
    label_test = label[:test_count]
    mask_test = mask[:test_count]
    extraVars_test={}
    for tag in extraVars:
        extraVars_test[tag]=extraVars[tag][:test_count]
    
    dataset =dataset[test_count:]
    label=label[test_count:]
    mask=mask[test_count:]
    for tag in extraVars:
        extraVars[tag]=extraVars[tag][test_count:]
        
    print("train dataset shape ",dataset.shape)
    print("test dataset shape ",data_test.shape)
    print("ASSERTING NO EVT LOSS  ",data_test.shape[0]+dataset.shape[0], "== ",allEvtCount)
    return {'data':dataset,'labels':label,'mask':mask,'extraVars':extraVars,
            'data_test':data_test,'label_test':label_test,'mask_test':mask_test,'extraVars_test':extraVars_test}


def getTestTrainDataLoaders(dataFileNames,inputVarList,labelVars,
                            isvalidMaskVar=None,otherVars={},
                            testTrainSplit=0.1,valTrainSplit=0.1,
                            nSigEvts=-1,nBkgEvts=-1,getDataLoader=True,evaluation=False):
    text="Loading Data"
    
    dataset=getDataset(dataFileNames,inputVarList,labelVars,
                       isvalidMaskVar=isvalidMaskVar,otherVars=otherVars,
                       nSigEvts=nSigEvts,nBkgEvts=nBkgEvts,
                       testTrainSplit=testTrainSplit,randomize=True)
    
    # ### Test Train/Validation Split
    
    text="Making Test train Split"
    printHead(text)
    
    data_train   = dataset['data']
    labels_train = dataset['labels']
    masks_train = dataset['mask']
    extraVars_train = dataset['extraVars']    
    
    print("data shape : ",data_train.shape)

    print("labels shape : ",labels_train.shape)
    print("Number of uniquie classes in training set : ",np.unique(labels_train))
    signalMask=( torch.sum(labels_train,dim=1) > 0  ).squeeze()
    bkgMask   =( torch.sum(labels_train,dim=1) < 0.5).squeeze()
    sorted_indices={}
    sorted_indices[1]=np.arange(labels_train.shape[0])[signalMask]
    sorted_indices[0]=np.arange(labels_train.shape[0])[bkgMask]
    
    
    print("Total number of events for training : ",labels_train.shape[0],"(",sum(signalMask),"+",sum(bkgMask),")")
    
    num_val_exmps = int(min(len(sorted_indices[0]),len(sorted_indices[1]))*valTrainSplit)
    print("Setting number of Vaidation elements as  ",num_val_exmps)
    
    # Get image indices for validation and training
    val_indices   = sorted_indices[0][:num_val_exmps]
    val_indices   = np.concatenate([val_indices,sorted_indices[1][:num_val_exmps]])
    
    train_indices = sorted_indices[0][num_val_exmps:]
    train_indices   = np.concatenate([train_indices,sorted_indices[1][num_val_exmps:]])
    
    # Group corresponding image features and labels
    train_feats   = data_train[train_indices]
    train_labels  = labels_train[train_indices]
    train_masks   = masks_train[train_indices]
    train_extraVars   = {tag: extraVars_train[tag][train_indices]  for tag in extraVars_train}
    val_feats  = data_train[val_indices]
    val_labels = labels_train[val_indices]
    val_masks  = masks_train[val_indices]
    val_extraVars  = {tag: extraVars_train[tag][val_indices]  for tag in extraVars_train}
    
    print("train_feats.shape : ",train_feats.shape)
    print("val_feats.shape   : ",val_feats.shape)
    
    print("ASSERT    : ",val_feats.shape[0]+train_feats.shape[0]," == ",labels_train.shape[0])
    
    test_feats = dataset['data_test']
    test_labels = dataset['label_test']
    test_masks = dataset['mask_test']
    test_extraVars = dataset['extraVars_test']

    if not getDataLoader:
        allDataDict={'data_train' : train_feats,'train_labels':train_labels,'train_extraVars':train_extraVars,
                'data_val' : val_feats,'val_labels':val_labels,'val_extraVars':val_extraVars,
                'data_test' : test_feats,'test_labels':test_labels,'test_extraVars':test_extraVars,
               }
        return allDataDict
    print("Is this an evaluation dataset : ",evaluation)
    train_dataset = trippleHDataset(train_feats, train_labels,train_masks,
                                    extraVals=train_extraVars, evaluation=evaluation)
    val_dataset   = trippleHDataset(val_feats  ,   val_labels,  val_masks,
                                    extraVals=val_extraVars, evaluation=evaluation)
    test_dataset  = trippleHDataset(test_feats ,  test_labels, test_masks,
                                    extraVals=test_extraVars, evaluation=evaluation)
    
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
    val_loader   = data.DataLoader(val_dataset,   batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader  = data.DataLoader(test_dataset,  batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    return {'train':train_loader,'val':val_loader,'test':test_loader}


def getDataLoaders(dataFileNames,inputVarList,labelVars,
                   isvalidMaskVar=None,otherVars={},
                   nEvts=-1,getDataLoader=True,evaluation=True):
    text="Loading Data"
    
    dataset=getDataset(dataFileNames,inputVarList,labelVars,
                       isvalidMaskVar=isvalidMaskVar,otherVars=otherVars,
                       nSigEvts=nEvts,nBkgEvts=nEvts,
                       testTrainSplit=0.0,randomize=False)
    
    # ### Test Train/Validation Split
    
    text="Making Test train Split"
    printHead(text)
    
    data_train   = dataset['data']
    labels_train = dataset['labels']
    masks_train = dataset['mask']
    extraVars_train = dataset['extraVars']    
    
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
    print("Setting number of Vaidation elements as  ",num_val_exmps)
    
    # Get image indices for validation and training
    val_indices   = sorted_indices[0][:num_val_exmps]
    val_indices   = np.concatenate([val_indices,sorted_indices[1][:num_val_exmps]])
    
    train_indices = sorted_indices[0][num_val_exmps:]
    train_indices   = np.concatenate([train_indices,sorted_indices[1][num_val_exmps:]])
    
    # Group corresponding image features and labels
    train_feats   = data_train[train_indices]
    train_labels  = labels_train[train_indices]
    train_masks   = masks_train[train_indices]
    train_extraVars   = {tag: extraVars_train[tag][train_indices]  for tag in extraVars_train}
    val_feats  = data_train[val_indices]
    val_labels = labels_train[val_indices]
    val_masks  = masks_train[val_indices]
    val_extraVars  = {tag: extraVars_train[tag][val_indices]  for tag in extraVars_train}
    
    print("train_feats.shape : ",train_feats.shape)
    print("val_feats.shape   : ",val_feats.shape)
    
    print("ASSERT    : ",val_feats.shape[0]+train_feats.shape[0]," == ",labels_train.shape[0])
    
    test_feats = dataset['data_test']
    test_labels = dataset['label_test']
    test_masks = dataset['mask_test']
    test_extraVars = dataset['extraVars_test']

    if not getDataLoader:
        allDataDict={'data_train' : train_feats,'train_labels':train_labels,'train_extraVars':train_extraVars,
                'data_val' : val_feats,'val_labels':val_labels,'val_extraVars':val_extraVars,
                'data_test' : test_feats,'test_labels':test_labels,'test_extraVars':test_extraVars,
               }
        return allDataDict
    print("Is this an evaluation dataset : ",evaluation)
    train_dataset = trippleHDataset(train_feats, train_labels,train_masks,
                                    extraVals=train_extraVars, evaluation=evaluation)
    val_dataset   = trippleHDataset(val_feats  ,   val_labels,  val_masks,
                                    extraVals=val_extraVars, evaluation=evaluation)
    test_dataset  = trippleHDataset(test_feats ,  test_labels, test_masks,
                                    extraVals=test_extraVars, evaluation=evaluation)
    
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
    val_loader   = data.DataLoader(val_dataset,   batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader  = data.DataLoader(test_dataset,  batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    return {'train':train_loader,'val':val_loader,'test':test_loader}




def train_hhhVsQCD(train_loader,val_loader,test_loader,
                   maxEpoch,version=None,
                   datasets=None,
                   tagsToFill=[],
                   CHECKPOINT_PATH='./chkpt.tmp',
                   savePathBase='./',
                   continueTraining=False,
                   mode='async',
                   with_id=None,
                   **kwargs):
    
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir='./tmp/'
    if not os.path.isfile(CHECKPOINT_PATH):
        root_dir = os.path.join(CHECKPOINT_PATH,'')
        tagsToFill.append("training")
    else:
        tagsToFill.append("eval")

    os.makedirs(root_dir, exist_ok=True)
    
    hpars=copy.deepcopy(kwargs)
    desc='single class descriminator'
    if 'remark' in desc:
        desc+=' , '+ hpars['remark']
    
    tb_logger = TensorBoardLogger(save_dir=root_dir)
    tb_logger.log_hyperparams(hpars)
    
    srcFiles=[
            'python/*.py',
            '../../python/*.py',
    ]
    if '1p0' in datasets['sig']['ggHHH']:
        tagsToFill.append('AllReco+Merged')
    elif '1p3' in datasets['sig']['ggHHH']:
        tagsToFill.append('AllReco')

    print("   with_id : ",with_id)
    print("      mode : ",mode)

    neptune_logger = NeptuneLogger(
        api_key=neptuneApiKey,  # 
        project=neptuneProject,
        tags=tagsToFill,  # optional
        description=desc,
        source_files=srcFiles,
        mode=mode,
        with_id=with_id
    )
    neptune_logger.experiment.sync()
    
    if datasets:
        if 'sig' in datasets:
            for ky in datasets['sig']:
                neptune_logger.experiment["dataset/sig/"+ky].track_files(datasets['sig'][ky])
            for ky in datasets['bkg']:
                neptune_logger.experiment["dataset/bkg/"+ky].track_files(datasets['bkg'][ky])
    
    neptune_logger.log_hyperparams(hpars)
    import neptune.new as neptune_new 
    nBatches=len(train_loader)
    loggers={'neptune_logger':neptune_logger,'tb_logger':tb_logger}
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=maxEpoch,
                         gradient_clip_val=2,
                         logger=[tb_logger,neptune_logger],
                         log_every_n_steps=int(0.05*nBatches))

    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    trainer.val_check_interval=0.05
    if 'inputVarList' in hpars:
        hpars['inputVarList']=str(hpars['inputVarList'])
    results={}



    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.abspath(CHECKPOINT_PATH)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = trippleHNonResonatModel.load_from_checkpoint(pretrained_filename)
        if continueTraining:
            trainer.fit(model, train_loader, val_loader)
    else:
        results={}
        print("Making Plots for input vars in tarin")

        train_loader.dataset.setEval(True)
        results['input_vars_train']=variablePlotter.plotAllVars(train_loader,'train',inputVarList,-1,savePrefix=savePathBase)
        train_loader.dataset.setEval(False)
     
        print("Making Plots for input vars in val")
        val_loader.dataset.setEval(True)
        results['input_vars_val']=variablePlotter.plotAllVars(val_loader,'val',inputVarList,-1,savePrefix=savePathBase)
        val_loader.dataset.setEval(False)
     
        print("Making Plots for input vars in test")
        test_loader.dataset.setEval(True)
        results['input_vars_test']=variablePlotter.plotAllVars(test_loader,'test',inputVarList,-1,savePrefix=savePathBase)
        test_loader.dataset.setEval(False)
     
        for tag in results:
            for ky in results[tag]:
                if 'plot' in ky:
                    loggers['neptune_logger'].experiment['inputvars/'+tag+'/'+ky].upload(results[tag][ky])
                elif 'fname' in ky:
                    loggers['neptune_logger'].experiment['inputvars/'+tag+'/'+ky].track_files(results[tag][ky])
    
                else:
                    loggers['neptune_logger'].experiment['inputvars/'+tag+'/'+ky]=results[tag][ky]
    
            model = trippleHNonResonatModel(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
            printHead("Traing the model for "+str(maxEpoch) +' epochs')
            trainer.fit(model, train_loader, val_loader)
            model = trippleHNonResonatModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result   = trainer.test(model, val_loader, verbose=False)
    test_result  = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"], "train_acc": train_result[0]["test_acc"]}
 #   model = model.to('cpu')
    return {'model':model,'trainer': trainer,'result':result,'loggers':loggers}

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

def trainBinaryClassifier(xTr,yTr,savePrefix='./'):
    n_hidden=3
    finalModel = torch.nn.Sequential(torch.nn.Linear(2, n_hidden),
                      torch.nn.ReLU(),
                      torch.nn.Linear(n_hidden, n_hidden),
                      torch.nn.ReLU(),
                      torch.nn.Linear(n_hidden, 1),
                      torch.nn.Sigmoid())
    learning_rate=0.1
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(finalModel.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(1000):
        x=xTr.detach()
        pred_y = finalModel(x)
        loss = loss_function(pred_y, yTr)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        finalModel.zero_grad()
    
    plt.figure()
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    fname_binaryClssificationTrainingLoss=savePrefix+'/binaryClssificationTrainingLoss'
    plt.savefig(fname_binaryClssificationTrainingLoss,bbox_inches='tight')
    
    return finalModel


if __name__=="__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Tag",default='tmp')
    parser.add_argument("--remark", help="Remark",default='Nothing specific ! LOL')
    parser.add_argument("--chk", help="Checkpoint",default=None)
    parser.add_argument("--resumeTraining", help="Checkpoint",default=0)
    parser.add_argument("--maxEpoch", help="Max Number of Epoches to run over",default=5)
    parser.add_argument("--logmode", help="Connection mode of neptune logger",default='async')
    parser.add_argument("--logTags", help="log tags",default='tmp')
    parser.add_argument("--with_id", help="Connec motune logger",default=None)
    args = parser.parse_args()
    
    tagsToFill=[]
    if args.logTags:
        ltags=args.logTags.split(',')
        for t in ltags:
            tagsToFill.append(t)

    if not args.chk:
        CHECKPOINT_PATH=('workarea/' +args.tag+'/trippleHiggsVsQCD/').replace('//','/')
    else:
        CHECKPOINT_PATH=args.chk
    print("CHECKPOINT_PATH : ",CHECKPOINT_PATH)
    tag=args.tag
    with_id=args.with_id
    maxEpoch=int(args.maxEpoch)
    log_mode=args.logmode
    if not os.path.isfile(CHECKPOINT_PATH): 
        root_dir=CHECKPOINT_PATH
    else:
        root_dir='workarea/extended/trippleHiggsVsQCD/'
    continueTraining=False
    if args.resumeTraining >0:
        continueTraining=True

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
        inputVarList.append('jet_'+str( i )+'_dEta_WithDP')
        inputVarList.append('jet_'+str( i )+'_dPhi_WithDP')
        inputVarList.append('jet_'+str( i )+'_mass_WithDP')
   
   #     inputVarList.append('jet_'+str( i )+'_drWithDP_leadG')
   #     inputVarList.append('jet_'+str( i )+'_dr_WithDPSubleadG')
   #     inputVarList.append('jet_'+str( i )+'_dEta_WithDPLeadG')
   #     inputVarList.append('jet_'+str( i )+'_dPhi_WithDPLeadG')
   #     inputVarList.append('jet_'+str( i )+'_mass_WithDPLeadG')

   #     inputVarList.append('jet_'+str( i )+'_dEta_WithDPSubleadG')
   #     inputVarList.append('jet_'+str( i )+'_dPhi_WithDPSubleadG')
   #     inputVarList.append('jet_'+str( i )+'_mass_WithDPSubleadG')
    
    
    labelVars=['label_'+str(i) for i in range(8)]
    isvalidMaskVar=['jet_'+str(i)+'_isValid' for i in range(8)]
    otherVars={ 
                    '4bIndex': {'var_names':['label_idx_'+str(i) for i in range(8)]}
           }
    
    print("Number of Input variables : ",len(inputVarList))
    print("Number of Input Labels : ",len(labelVars))
    
    dataFileNames={
        'sig':{
            'ggHHH' : '/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018//ggHHH/ml_ggHHH_upd_3p0_part1.root'
        },
        'bkg':{
            'ggM80Jbox2bjet' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton2BJets_3p02.root',
            'ggM80Jbox1bjet' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton1BJets_3p02.root',
            'ggM80Inc' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton_3p02.root'
        }
    }

    dataLoaders = getTestTrainDataLoaders(dataFileNames,inputVarList,labelVars,
                                          isvalidMaskVar=isvalidMaskVar,otherVars=otherVars,
                                          testTrainSplit=0.2,valTrainSplit=0.1,
                                          nSigEvts=-1,nBkgEvts=-1)
                                          
    results={}
    savePathBase = root_dir+'/modelScratch/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S').replace('-','_')+'_'+str(uuid.uuid4().hex)+'/'
    os.system('mkdir -p '+savePathBase)
    print('Base directory for evaluation/training metrics : ',savePathBase)
    
    train_loader = dataLoaders['train']
    val_loader = dataLoaders['val']
    test_loader = dataLoaders['test']
    train_sample,mask,_=train_loader.dataset[0];


    print("Tarining The model !!  ",train_sample[-1].shape)

    trainingResult = train_hhhVsQCD( train_loader , val_loader,test_loader,
                                                  maxEpoch=maxEpoch,
                                                  inputVarList=inputVarList,
                                                  remark=tag + args.remark,
                                                  mode=log_mode,
                                                  CHECKPOINT_PATH=CHECKPOINT_PATH,
                                                  continueTraining=continueTraining,
                                                  tagsToFill=tagsToFill,
                                                  with_id=with_id,
                                                  datasets=dataFileNames,
	    				                          input_dim=int(len(inputVarList)/8),
                                                  model_dim=64,
                                                  num_heads=8,
                                                  num_classes=2,
                                                  num_layers=2,
                                                  dropout=0.1,
                                                  input_dropout=0.0,
                                                  lr=5e-4,
                                                  warmup=200,
                                                  savePathBase=savePathBase
                                                )
    hhhVsQCD_model  = trainingResult['model']
    model_trainer   = trainingResult['trainer']
    hhhVsQCD_result = trainingResult['result']
    loggers=trainingResult['loggers']

    hhhVsQCD_model.eval()
    
    print("Training Accuracy    : ",hhhVsQCD_result['train_acc'])
    print("Validation  Accuracy : ",hhhVsQCD_result['val_acc'])
    print("Testing  Accuracy    : ",hhhVsQCD_result['test_acc'])

    for tag in dataLoaders:
        dataLoaders[tag].dataset.setEval(True)
    hhhVsQCD_model=hhhVsQCD_model.to(device)
    print("Model is reciding in  ",hhhVsQCD_model.device)

    evalResult={}
    for tag in dataLoaders:
        print("Loading evaluation results for  ",tag)
        evalResult[tag]=modelEval.getEvaluationResults(dataLoaders[tag],hhhVsQCD_model,bMax=-1)
    

    hhhTo4bMask={}
    for tag in evalResult:
        hhhTo4bMask[tag]={}
        for i in range(1,4+1):
            hhhTo4bMask[tag]['4bMask_'+str(i)]= evalResult[tag]['4bIndex']==i
        hhhTo4bMask[tag]['signal']=evalResult[tag]['label']==1
        hhhTo4bMask[tag]['background']=evalResult[tag]['label']!=1


    for tag in ['test','train']:
        results['transformer_model_'+tag]=modelEval.saveSignalVsBkg_Y0Y1plots(evalResult[tag],hhhTo4bMask[tag],tag=tag,savePrefix=savePathBase)   

    ###  training the binary calassifier
    
    trainX=evalResult['train']['yPred'][evalResult['train']['vldMask']]
    trainY=evalResult['train']['label'][evalResult['train']['vldMask']].float()
    trainY=trainY.view(-1,1)

    testX=evalResult['test']['yPred'][evalResult['test']['vldMask']]
    testY=evalResult['test']['label'][evalResult['test']['vldMask']].float()
    testY=testY.view(-1,1)
    
    print("Training The Binary Classifier !! ")
    binaryClsModel=trainBinaryClassifier(trainX,trainY,savePrefix=savePathBase)
    print("Analyzing the Binary Classifier !! ")
    results['binaryCls']=modelEval.saveBinaryClassifier_PerformancePlots(binaryClsModel,trainX,trainY,testX,testY,savePrefix=savePathBase)
    threshold=results['binaryCls']['threshold']
    print("Saving the Binary Classifier !! ")
    binaryClsModelFilename=savePathBase+'binaryClasifcationModel.pkl'
    pickle.dump(binaryClsModel, open(binaryClsModelFilename, 'wb'))
    results['binaryCls']['classifier_fname']=binaryClsModelFilename
    
    print("Performance monitoring of Binary Classifier !! ")
    for tag in ['train','test']:
        results['binaryCls'+'_'+tag]=modelEval.saveSignalVsBkg_binaryEventLevelPlots(binaryClsModel,evalResult[tag],thr=threshold,savePrefix=savePathBase)
    
    for tag in results:
        head=tag.replace('_test','').replace('_train','')
        sub='eval'
        if 'test' in tag:
            sub='test'
        if 'train' in tag:
            sub='train'
        print(head,sub)
        for ky in results[tag]:
            if 'plot' in ky:
                loggers['neptune_logger'].experiment[head+'/'+sub+'/'+ky].upload(results[tag][ky])
            elif 'fname' in ky:
                loggers['neptune_logger'].experiment[head+'/'+sub+'/'+ky].track_files(results[tag][ky])

            else:
                loggers['neptune_logger'].experiment[head+'/'+sub+'/'+ky]=results[tag][ky]
