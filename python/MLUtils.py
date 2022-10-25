import uproot3 as upr
import pandas as pd
import torch

def printHead(text):
    n=len(text)
    print("\n","- "*n,"\n",text,"\n","- "*n)

def readDataset(dataFileNames,labelVars,treeName='bkg_13TeV_TrippleHTag_0_background',inputVarList=[],NEVTS=-1,N_JETS=8):
    outputDset=None
    labels=None
    print("Londing ", NEVTS,"events per file ")
    for tag in dataFileNames:
        print("Loading  : ",tag)
        print("     file : ",dataFileNames[tag])
        f=upr.open(dataFileNames[tag])
        inputTree=f['trees'][treeName]
        inputTreeVars=[ i.decode("utf-8") for  i in inputTree.keys() ]
        
       # print("input var list \n",inputVarList)
       # print("\ninput var list \n",inputTreeVars)
        
        for var in inputVarList:
            if var not in inputTreeVars:
                print('\t : ',var," variable not in the tree ",treeName," in file ",dataFileNames[tag])
                
        dataDict=inputTree.arrays(inputVarList)
        inputData=pd.DataFrame.from_dict(dataDict).to_numpy()[:NEVTS]
        nCount=inputData.shape[0]
        for var in labelVars:
            if var not in inputTreeVars:
                print('\t : ',var," label not in the tree ",treeName," in file ",dataFileNames[tag])
            
        labelDict=inputTree.arrays(labelVars)
        labelData=pd.DataFrame.from_dict(labelDict).to_numpy(dtype='int')[:NEVTS]
        
        print("    evts  : ",nCount)
        inputData=torch.tensor(inputData)
        inputData=inputData.reshape(nCount,N_JETS,-1)
        labelData=torch.LongTensor(labelData)
        labelData=labelData.reshape(nCount,-1,N_JETS)
        if outputDset==None:
            outputDset=inputData
            labels=labelData
        else:
            outputDset=torch.concat([outputDset,inputData],0)
            labels=torch.concat([labels,labelData ],0)
        f.close()
        
    return outputDset,labels


