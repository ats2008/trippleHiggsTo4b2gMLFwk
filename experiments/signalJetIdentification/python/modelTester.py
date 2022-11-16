#!/usr/bin/env python
# coding: utf-8

import argparse,copy

import uproot3 as upr

import pandas as pd

import torch
import torch.utils.data as tData
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

import modelEval,modelDev,variablePlotter
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

def getDataset(dataFileNames,inputVarList,labelVars,treeName,
               isvalidMaskVar=None,otherVars={},
               seedVal=42,nEvts=-1):

    dset=mlUtil.readDataset(dataFileNames,labelVars,
                                          treeName,
                                          inputVarList,otherVars=otherVars,
                                          isvalidMaskVar=isvalidMaskVar,
                                          NEVTS=nEvts)
    dataset   = dset['data']
    label     = dset['label']
    mask      = dset['mask']
    extraVars = dset['otherVars']

    Mem=sys.getsizeof(dataset.storage())/1e6
    extraVarMem=sum([sys.getsizeof(extraVars[tag].storage()) for tag in extraVars])/1e6

    print("Total number of Events : ",dataset.shape[0]," [ ",Mem," MB]")
    print("Total memory occupied by extra var : ",extraVarMem," MB]")

        
    return {'data':dataset,'labels':label,'mask':mask,'extraVars':extraVars}




def getDataLoaders(dataFileNames,inputVarList,labelVars,treeName,
                   isvalidMaskVar=None,otherVars={},
                   nEvts=-1,evaluation=True):
    text="Loading Data"
    
    dataset=getDataset(dataFileNames,inputVarList,labelVars,treeName,
                       isvalidMaskVar=isvalidMaskVar,otherVars=otherVars,
                       nEvts=nEvts)
    
    # ### Test Train/Validation Split
    
    text="Making Test train Split"
    printHead(text)
    
    data   = dataset['data']
    labels = dataset['labels']
    masks  = dataset['mask']
    extraVars = dataset['extraVars']    
    
    print("Is this an evaluation dataset : ",evaluation)
    
    dataset = trippleHDataset(data, labels,masks,extraVars, evaluation=evaluation)
    data_loader = tData.DataLoader(dataset, batch_size=128, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)

    return data_loader


def load_binaryClasiifier(CHECKPOINT_PATH='./chkpt.tmp'):
    # Create a PyTorch Lightning trainer with the generation callback
    if not os.path.exists(CHECKPOINT_PATH):
        raise Exception(CHECKPOINT_PATH+" do not exists ! ")
    
    if os.path.isfile(CHECKPOINT_PATH):
        print("Found pretrained model, loading...")
        TODO
        model = trippleHNonResonatModel.load_from_checkpoint(CHECKPOINT_PATH)
        model.eval()
    else:
        raise Exception(CHECKPOINT_PATH+" is not a file ! ")
        
    return model

if __name__=="__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Tag",default='tmp')
    parser.add_argument("--logTags", help="log tags",default='tmp')
    parser.add_argument("--remark", help="Remark",default='Nothing specific ! LOL')
    parser.add_argument("--chk", help="Checkpoint",default=None)
    parser.add_argument("--threshold", help="Binary classifiction threshold",default=0.5)
    parser.add_argument("--maxBatches", help="Max Batches to analyze ",default=-1)
    parser.add_argument("--binaryClsModel", help="binary Clsssification Model path ",default='')
    parser.add_argument("--logmode", help="Connection mode of neptune logger",default='async')
    parser.add_argument("--with_id", help="Connec motune logger",default=None)
    args = parser.parse_args()
    
    if not args.chk:
        CHECKPOINT_PATHa=('workarea/' +args.tag).replace('//','/')
    else:
        CHECKPOINT_PATH=args.chk
    root_dir=('workarea/' +args.tag+'/trippleHiggsVsQCD').replace('//','/')
    if not os.path.isfile(CHECKPOINT_PATH):
        root_dir = os.path.join(CHECKPOINT_PATH, "trippleHiggsVsQCD")
    print("CHECKPOINT_PATH : ",CHECKPOINT_PATH)
    tag=args.tag
    with_id=args.with_id
    mode=args.logmode
    binaryClasifcationModelFileName=args.binaryClsModel
    threshold=args.threshold
    maxBatches=int(args.maxBatches)
    print("Max Batches : ", maxBatches)

    desc='single class descriminator evaluation'
    if 'remark' in desc:
        desc+=' , '+ hpars['remark']
    
    print("Analyzing ",maxBatches," batches")


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
        'hhhV3p02':{
            'ggHHH' : '/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/ggHHH/ml_ggHHH_upd_3p3p1.root'
        },
        'hhhV3p0_fullMC':{
            'ggHHH' : '/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/ggHHH/ml_ggHHH_upd_full_3p0.root'
        },
        'qcd2b':{
            'ggM80Jbox2bjet' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton2BJets_3p02.root'
        },
        'qcd1b':{
            'ggM80Jbox1bjet' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton1BJets_3p02.root'
        },
        'qcdInc':{
            'ggM80Inc' :'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/diphotonX/ml_diPhoton_3p02.root'
        },
        'data2018A':{
            'data2018A':'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/data/2018A/ml_data2018A_3p02.root',
        },
        'data2018B':{
            'data2018B':'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/data/2018B/ml_data2018B_3p02.root',
        },
        'data2018C':{
            'data2018C':'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/data/2018C/ml_data2018C_3p02.root',
        },
        'data2018D':{
            'data2018D':'/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/data/2018D/ml_data2018D_3p02.root',
        }
    }

    dataTreeNames={
        'hhhV3p02':'ggHHH_125_13TeV_allRecoed',
        'hhhV3p0_fullMC':'ggHHH_125_13TeV_allEvents',
        'qcd2b':'bkg_13TeV_TrippleHTag_0_background',
        'qcd1b':'bkg_13TeV_TrippleHTag_0_background',
        'qcdInc':'bkg_13TeV_TrippleHTag_0_background',
        'data2018A':'Data_13TeV_TrippleHTag_0_background',
        'data2018B':'Data_13TeV_TrippleHTag_0_background',
        'data2018C':'Data_13TeV_TrippleHTag_0_background',
        'data2018D':'Data_13TeV_TrippleHTag_0_background'
    }
    
    results={}
    savePathBase = root_dir+'/modelScratch/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S').replace('-','_')+'_'+str(uuid.uuid4().hex)+'/'
    print("Log Scratch Dir : ",savePathBase)
    os.system('mkdir -p '+savePathBase)
    print('Base directory for evaluation metrics : ',savePathBase)


    dataLoaders={}
    for kyA in dataFileNames:
        dataLoaders[kyA] = getDataLoaders(dataFileNames[kyA],inputVarList,labelVars,dataTreeNames[kyA],
                                      isvalidMaskVar=isvalidMaskVar,otherVars=otherVars,
                                      nEvts=-1,evaluation=True)
    for tag in dataLoaders:
        dataLoaders[tag].dataset.setEval(True)

    hhhVsQCD_model  = modelDev.load_model(CHECKPOINT_PATH).to(device)
    hhhVsQCD_model.eval()
    
    srcFiles=[
            'python/*.py',
            '../../python/*.py',
    ]
    
    evalPrefix='eval/'

    print("   with_id : ",with_id)
    print("      mode : ",mode)
    tagsToFill=['eval']
    if args.logTags:
        ltags=args.logTags.split(',')
        for t in ltags:
            tagsToFill.append(t)

    for kyA in dataFileNames:
        for kyB in dataFileNames[kyA]:
            if 'ggHHH' in kyA:
                if '1p0' in dataFileNames[kyA][kyB]:
                     tagsToFill.append('AllReco+Merged')
                elif '1p3' in datasets['sig']['ggHHH']:
                    tagsToFill.append('AllReco')

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

    for kyA in dataFileNames:
        for kyB in dataFileNames[kyA]:
            neptune_logger.experiment[evalPrefix+"/dataset/"+kyA+'_'+kyB].track_files(dataFileNames[kyA][kyB])
 
    loggers={'neptune_logger':neptune_logger}
    
    for tag in dataLoaders:
        loggers['neptune_logger'].experiment[evalPrefix+'/'+tag+'_scratchArea'] =savePathBase   
        dataLoaders[tag].dataset.setEval(True)
        results['vars_'+tag]=variablePlotter.plotAllVars(dataLoaders[tag],tag,inputVarList,bMax=maxBatches,savePrefix=savePathBase)
     
    hhhVsQCD_model=hhhVsQCD_model.to(device)
    print("Model is reciding in  ",hhhVsQCD_model.device)

    evalResult={}
    for tag in dataLoaders:
        print("Loading evaluation results for  ",tag)
        evalResult[tag]=modelEval.getEvaluationResults(dataLoaders[tag],hhhVsQCD_model,bMax=maxBatches)

    hhhTo4bMask={}
    for tag in evalResult:
        hhhTo4bMask[tag]={}
        for i in range(1,4+1):
            hhhTo4bMask[tag]['4bMask_'+str(i)]= evalResult[tag]['4bIndex']==i
        hhhTo4bMask[tag]['signal']=evalResult[tag]['label']==1
        hhhTo4bMask[tag]['background']=evalResult[tag]['label']!=1

    for tag in evalResult:
        results['transformer_model_'+tag]=modelEval.saveEval_Y0Y1plots( evalResult[tag] , hhhTo4bMask[tag],tag=tag,savePrefix=savePathBase)   

    results['modelFiles']={
        'CHECKPOINT_PATH' : CHECKPOINT_PATH,
        'binaryClasifcationModelFileName' : binaryClasifcationModelFileName
        }

    if binaryClasifcationModelFileName!='':
        print("Opening the Binary Classifier !!  fname : ",binaryClasifcationModelFileName)
        binaryClsModel=pickle.load(open(binaryClasifcationModelFileName, 'rb'))
        for tag in evalResult:
            print(" making the Perfomance plots for : ",tag)
            results['binaryCls_'+tag]=modelEval.saveBinaryClassifier_PerformanceEvalPlots(binaryClsModel,evalResult[tag],hhhTo4bMask[tag],tag=tag,savePrefix=savePathBase)
        
        print("Performance monitoring of Binary Classifier !! ")
        for tag in evalResult:
            print(" making the Event level Perfomance plots for : ",tag)
            results['binaryCls'+'_'+tag].update(modelEval.saveSignalVsBkg_binaryEventLevelEvalPlots(binaryClsModel,evalResult[tag],tag=tag,thr=threshold,savePrefix=savePathBase))

    for tag in results:
        for ky in results[tag]:
            if 'plot' in ky:
                loggers['neptune_logger'].experiment[evalPrefix+tag+'/'+ky].upload(results[tag][ky])
            else:
                loggers['neptune_logger'].experiment[evalPrefix+tag+'/'+ky]=results[tag][ky]

