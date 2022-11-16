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

import pickle as pkl
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
    parser.add_argument("--chk", help="Checkpoint",default=None)
    parser.add_argument("--maxBatches", help="Max Batches to analyze ",default=-1)
    parser.add_argument("--binaryClsModel", help="binary Clsssification Model path ",default='')
    parser.add_argument("--outDir", help="output directory for pickled exports",default='./')
    parser.add_argument("--pklProtocol", help="output protocol for pickled exports",default=pkl.HIGHEST_PROTOCOL)
    args = parser.parse_args()

    pklProtocol=int(args.pklProtocol)
    print("Protocol for pickle  ",pklProtocol)
    savePathBase=args.outDir
    os.system('mkdir -p '+savePathBase)
    
    CHECKPOINT_PATH=args.chk
    if not os.path.isfile(CHECKPOINT_PATH):
        print("Checkpoint file not found : ",CHECKPOINT_PATH)
        print("exiting !")
        exit()

    print("CHECKPOINT_PATH : ",CHECKPOINT_PATH)
    
    binaryClasifcationModelFileName=args.binaryClsModel
    if not os.path.isfile(binaryClasifcationModelFileName):
        print("Binary Classification file not found : ",binaryClasifcationModelFileName)
        print("exiting !")
        exit()
    
    maxBatches=int(args.maxBatches)
    print("Max Batches : ", maxBatches)

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
                    '4bIndex': {'var_names':['label_idx_'+str(i) for i in range(8)]},
                    'eventIndex': {'var_names':['event']}
           }
    
    print("Number of Input variables : ",len(inputVarList))
    print("Number of Input Labels : ",len(labelVars))
    
    dataFileNames={
        'hhhV3p0_fullMC':{
            'ggHHH' : '/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/ggHHH/ml_ggHHH_upd_full_3p0.root'
        },
        'hhhV3p02':{
            'ggHHH' : '/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/ggHHH/ml_ggHHH_upd_3p3p1.root'
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
    dataFileNames={
        'hhhV3p0_fullMC':{
            'ggHHH' : '/grid_mnt/t3storage3/asugunan/store/trippleHiggs/mlNtuples/MC/2018/ggHHH/ml_ggHHH_upd_full_3p0.root'
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
    


    print("Loading the Tranformer model !!  fname : ",CHECKPOINT_PATH)
    hhhVsQCD_model  = modelDev.load_model(CHECKPOINT_PATH).to(device)
    hhhVsQCD_model.eval()
    print("Loading the Binary Classifier !!  fname : ",binaryClasifcationModelFileName)
    binaryClsModel=pkl.load(open(binaryClasifcationModelFileName, 'rb'))

    dataLoaders={}
    for kyA in dataFileNames:
        tag=kyA
        print("Loading evaluation results for  ",tag)
        
        dataLoader = getDataLoaders(dataFileNames[kyA],inputVarList,labelVars,dataTreeNames[kyA],
                                      isvalidMaskVar=isvalidMaskVar,otherVars=otherVars,
                                      nEvts=-1,evaluation=True)
        evalResult=modelEval.getEvaluationResults(dataLoader,hhhVsQCD_model,bMax=maxBatches)
        pred_y = binaryClsModel(evalResult['yPred'])
        pred_y[torch.logical_not(evalResult['vldMask'])]= -2.0
        result={}
        result['eventIndex']=np.asarray(evalResult['eventIndex'].detach().numpy(),dtype=int)
        result['vldMMask']=evalResult['vldMask'].detach().numpy()
        result['label']   =evalResult['label'].detach().numpy()
        result['4bIndex'] =evalResult['4bIndex'].detach().numpy()
        result['y0']=evalResult['yPred'][:,:,0].detach().numpy()
        result['y1']=evalResult['yPred'][:,:,1].detach().numpy()
        result['score']=pred_y.detach().numpy()
        result['modelCheckpoint']=CHECKPOINT_PATH
        result['binaryClsModel']=binaryClasifcationModelFileName
        result['dataset']=dataFileNames[kyA]

        fname=savePathBase+'/'+'mlOutPuts_'+tag+'.pkl'
        pkl.dump(result, open(fname, 'wb'),protocol=pklProtocol)

        print(tag,"  :  output file  : ",fname)
