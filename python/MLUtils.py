import uproot3 as upr
import pandas as pd
import torch

def printHead(text):
    n=len(text)
    print("\n","- "*n,"\n",text,"\n","- "*n)

def readDataset(dataFileNames,label=1,treeName='bkg_13TeV_TrippleHTag_0_background',inputVarList=[],NEVTS=-1,N_JETS=8):
    outputDset=None
    labels=None
    print("Londing ", NEVTS,"events per file ")
    for tag in dataFileNames:
        print("Loading  : ",tag)
        print("     file : ",dataFileNames[tag])
        f=upr.open(dataFileNames[tag])
        inputTree=f['trees'][treeName]
        dataDict=inputTree.arrays(inputVarList)
        inputData=pd.DataFrame.from_dict(dataDict).to_numpy()[:NEVTS]
        nCount=inputData.shape[0]
        print("    evts  : ",nCount)
        inputData=inputData.reshape(nCount,N_JETS,-1)
        inputData=torch.tensor(inputData)
        inputData=inputData.reshape(nCount,N_JETS,-1)
        if outputDset==None:
            outputDset=inputData
            labels=torch.LongTensor([label]*nCount )
        else:
            outputDset=torch.concat([outputDset,inputData],0)
            labels=torch.concat([labels,torch.LongTensor([label]*nCount )],0)
        f.close()
        
    return outputDset,labels




