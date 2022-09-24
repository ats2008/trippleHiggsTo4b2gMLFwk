#!/usr/bin/env python
# coding: utf-8

# In[24]:


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
from sklearn.metrics import roc_auc_score


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


# In[3]:


inputVarList=["jet_0_pt","jet_0_eta","jet_0_phi","jet_0_mass","jet_0_deepCSVScore","jet_0_bJetRegCorr","jet_0_bJetRegRes",
              "jet_1_pt","jet_1_eta","jet_1_phi","jet_1_mass","jet_1_deepCSVScore","jet_1_bJetRegCorr","jet_1_bJetRegRes",
              "jet_2_pt","jet_2_eta","jet_2_phi","jet_2_mass","jet_2_deepCSVScore","jet_2_bJetRegCorr","jet_2_bJetRegRes",
              "jet_3_pt","jet_3_eta","jet_3_phi","jet_3_mass","jet_3_deepCSVScore","jet_3_bJetRegCorr","jet_3_bJetRegRes",
              "jet_4_pt","jet_4_eta","jet_4_phi","jet_4_mass","jet_4_deepCSVScore","jet_4_bJetRegCorr","jet_4_bJetRegRes",
              "jet_5_pt","jet_5_eta","jet_5_phi","jet_5_mass","jet_5_deepCSVScore","jet_5_bJetRegCorr","jet_5_bJetRegRes",
              "jet_6_pt","jet_6_eta","jet_6_phi","jet_6_mass","jet_6_deepCSVScore","jet_6_bJetRegCorr","jet_6_bJetRegRes",
              "jet_7_pt","jet_7_eta","jet_7_phi","jet_7_mass","jet_7_deepCSVScore","jet_7_bJetRegCorr","jet_7_bJetRegRes",
              ]
print(len(inputVarList))



dataFileNames={
    'sig':{
        'ggHHH':'workarea/batch/ntupleForML/ggHHHto4b2gamma_UL17_13TeV_v2.root'
    },
    'bkg':{
        'ggM80Inc':'workarea/batch/ntupleForMLBkg/DiPhotonJetsBox_MGG-80toInf_13TeV-sherpa.root',
#         'ggM80Jbox1bjet':
        'ggM80Jbox2bjet':'workarea/batch/ntupleForMLBkg/DiPhotonJetsBox2BJets_MGG-80toInf_13TeV-sherpa.root'
#         'gJet20To40'  :
#         'gJet40ToInf' :
    }
}



def getDataset(dataFileNames,testTrainSplit=0.1,randomize=True,seedVal=42):
    bkgDataset,sigLabl=mlUtil.readDataset(dataFileNames['bkg'],0,'bkg_13TeV_TrippleHTag_0_background',inputVarList,NEVTS=20000)
    sigDataset,bkgLabl=mlUtil.readDataset(dataFileNames['sig'],1,'ggHHH_125_13TeV_allRecoed',inputVarList,NEVTS=-1)
    dataset=torch.cat([bkgDataset,sigDataset],0)
    label=torch.cat([bkgLabl,sigLabl],0)
    bkgMem=sys.getsizeof(bkgDataset.storage())/1e6
    sigMem=sys.getsizeof(sigDataset.storage())/1e6
    print("Total number of Signal Events     : ",sigDataset.shape[0]," [ ",sigMem," MB ]")
    print("Total number of Background Events : ",bkgDataset.shape[0]," [ ",bkgMem," MB]")
    if True : #randomize:
        print("   randomizing the dataset ! (with numpy seed : ",seedVal,")")
        np.random.seed(seedVal)
        permut=np.random.permutation(dataset.shape[0])
        dataset=dataset[permut]
        label=label[permut]
    test_count = int(min(sum(label==0),sum(label==1))/5)
    data_test   = dataset[:test_count]
    label_test = label[:test_count]
    dataset =dataset[test_count:]
    label=label[test_count:]
    print("train dataset shape ",dataset.shape)
    print("test dataset shape ",data_test.shape)
    return {'data':dataset,'labels':label,'data_test':data_test,'label_test':label_test}


text="Loading Data"

dataset=getDataset(dataFileNames,False)

# ### Test Train/Validation Split

text="Making Test train Split"
printHead(text)

data_train   = dataset['data']
labels_train = dataset['labels']


sorted_indices=[]
for i in [0,1]:
    sorted_indices.append(np.arange(labels_train.shape[0])[labels_train==i])

# sorted_indices=np.array(sorted_indices)
# Determine number of validation images per class


num_val_exmps = int(min(sum(labels_train==0),sum(labels_train==1))/10)
print("Setting number of Vaidation elements as  ",num_val_exmps)
# Get image indices for validation and training
val_indices   = sorted_indices[0][:num_val_exmps]
val_indices   = np.concatenate([val_indices,sorted_indices[1][:num_val_exmps]])

train_indices = sorted_indices[0][num_val_exmps:]
train_indices   = np.concatenate([train_indices,sorted_indices[1][num_val_exmps:]])

# Group corresponding image features and labels
train_feats, train_labels = data_train[train_indices], labels_train[train_indices]
val_feats,   val_labels   = data_train[val_indices],   labels_train[val_indices]

print("train_feats.shape : ",train_feats.shape)
print("val_feats.shape   : ",val_feats.shape)


# ## Dataset Defenition

# In[8]:


class trippleHDataset(data.Dataset):
    def __init__(self, features, labels, train=True):
        """
        Inputs:
            features - Tensor of shape [num_evts, evt_dim]. Represents the high-level features.
            labels - Tensor of shape [num_evts], containing the class labels for the events
            train - If True nothing will happen
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.train = train

        # Tensors with indices of the images per class
        self.num_labels = labels.max()+1

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # We return the indices of the event for visualization purpose."Label" is the class
        return self.features[idx], idx, self.labels[idx]


# In[9]:


test_feats = dataset['data_test']
test_labels = dataset['label_test']

train_dataset = trippleHDataset(train_feats, train_labels, train=True)
val_dataset   = trippleHDataset(val_feats  ,   val_labels, train=False)
test_dataset  = trippleHDataset(test_feats ,  test_labels, train=False)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
val_loader   = data.DataLoader(val_dataset,   batch_size=64, shuffle=False, drop_last=False, num_workers=4)
test_loader  = data.DataLoader(test_dataset,  batch_size=64, shuffle=False, drop_last=False, num_workers=4)


# In[ ]:





# ## Model Defenition

from TransformerModel import *


# Needed for initializing the lr scheduler
p = nn.Parameter(torch.empty(4,4))
optimizer = optim.Adam([p], lr=1e-3)
lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)

# Plotting
# epochs = list(range(2000))
# sns.set()
# plt.figure(figsize=(8,3))
# plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
# plt.ylabel("Learning rate factor")
# plt.xlabel("Iterations (in batches)")
# plt.title("Cosine Warm-up Learning Rate Scheduler")
# plt.show()
# sns.reset_orig()


# In[35]:


class trippleHNonResonatModel(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        features, _, labels = batch
        preds = self.forward(features, add_positional_encoding=False) # No positional encodings as it is a set, not a sequence!
        preds = preds.squeeze(dim=-1) # Shape: [Batch_size, set_size]
        loss = F.cross_entropy(preds, labels) # Softmax/CE over set dimension
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        #auc = roc_auc_score(  labels  , preds  )
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True)
        #self.log(f"{mode}_roc", auc, on_step=False, on_epoch=True)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


# ### training the Model

# In[18]:


CHECKPOINT_PATH="workarea/ml/attention4HHH/checkpoints/"


# In[28]:


def train_hhhVsQCD(**kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "trippleHiggsVsQCD")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=300,
                         gradient_clip_val=2)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "trippleHiggsVsQCD.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = trippleHNonResonatModel.load_from_checkpoint(pretrained_filename)
    else:
        model = trippleHNonResonatModel(max_iters=trainer.max_epochs*len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = trippleHNonResonatModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result   = trainer.test(model, val_loader, verbose=False)
    test_result  = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"], "train_acc": train_result[0]["test_acc"]}

    model = model.to(device)
    return model, result


# In[36]:

hhhVsQCD_model, hhhVsQCD_result = train_hhhVsQCD(input_dim=train_feats.shape[-1],
                                              model_dim=64,
                                              num_heads=8,
                                              num_classes=1,
                                              num_layers=4,
                                              dropout=0.1,
                                              input_dropout=0.0,
                                              lr=5e-4,
                                              warmup=200)


