import matplotlib.pyplot as plt 
import modelTester
import numpy as np
import torch

def getDatsetAsTensor(dataLoader,bMax=-1):
    dataStores={}
    with torch.no_grad():
        bCount=0
        for batch in iter(dataLoader):
            
            if bCount%50==0:
                print("\t Processing batch ",bCount," [ max : ",bMax," ] ")
            if bMax>0 and bCount>=bMax:
                break
            x,mask,label,extraVars=batch
            if 'data' not in dataStores:
                dataStores['data']=x
            else:
                dataStores['data']=torch.concat([dataStores['data'],x],dim=0)
            if 'mask' not in dataStores:
                dataStores['mask']=mask
            else:
                dataStores['mask']=torch.concat([dataStores['mask'],mask],dim=0)
            if 'label' not in dataStores:
                dataStores['label']=label
            else:
                dataStores['label']=torch.concat([dataStores['label'],label],dim=0)
            if 'extraVars' not in dataStores:
                dataStores['extraVars']={}
                for ky in extraVars:
                    dataStores['extraVars'][ky]=extraVars[ky]
            else:
                for ky in extraVars:
                    dataStores['extraVars'][ky]=torch.concat([dataStores['extraVars'][ky],
                                                                   extraVars[ky]],dim=0)
            bCount+=1
                    
    return dataStores

binDict={
    'jet_pt' : np.arange(0.0,500.0,5.0),
    'jet_eta' : np.arange(-4.0,4.0,0.10),
    'jet_phi' :  np.arange(-4.0,4.0,0.10),
    'jet_mass' :  np.arange(0.0,80.0,2.0),
    'jet_deepCSVScore' :np.arange(-2.2,2.0,0.10), 
    'jet_bJetRegCorr' : np.arange(0.0,2.0,0.05), 
    'jet_bJetRegRes' : np.arange(0.0,2.0,0.05), 
    'jet_drWithDP' : np.arange(0.0,4.0,0.05), 
    'jet_drWithDP_leadG' : np.arange(0.0,4.0,0.05), 
    'jet_dr_WithDPSubleadG' : np.arange(0.0,4.0,0.05), 
    'jet_dEta_WithDP' : np.arange(0.0,4.0,0.05), 
    'jet_dEta_WithDPLeadG' : np.arange(0.0,4.0,0.05), 
    'jet_dEta_WithDPSubleadG' : np.arange(0.0,4.0,0.05), 
    'jet_dPhi_WithDP' : np.arange(-4.0,4.0,0.05), 
    'jet_dPhi_WithDPLeadG' : np.arange(-4.0,4.0,0.05), 
    'jet_dPhi_WithDPSubleadG' : np.arange(-4.0,4.0,0.05), 
    'jet_mass_WithDP' :np.arange(0.0,1000.0,5.0),
    'jet_mass_WithDPLeadG' : np.arange(0.0,1000.0,5.0),
    'jet_mass_WithDPSubleadG' : np.arange(0.0,1000.0,5.0),
}

def plotPtEtaPhiM(dataset,mask,varMap={'jet_pt':0,'jet_eta':1,'jet_phi':2,'jet_mass':3},savePrefix='./',tag='',jetIdx=0):
    f,ax=plt.subplots(nrows=2,ncols=2)
    ax=np.ndarray.flatten(ax)
    i=0
    for var in varMap:
        n=varMap[var]
        _=ax[i].hist(dataset[n][jetIdx][mask],bins=binDict[var],histtype='step')
        i+=1
    fname1=savePrefix+'/'+tag+'_pTetaPhiM_jet'+str(jetIdx)+'.png'
    f.savefig(fname1)
    
    return {'plot_ptEtaPhiM_jet'+str(jetIdx):fname1 }

def plotInputVars( dataset , qcdMask ,jetMask , varMap ,tag ,prefix='./'):
    result={}
    for var in varMap:
               
        f,ax=plt.subplots(nrows=1,ncols=1)
        n=varMap[var]

        bkgVals=dataset[n][qcdMask].flatten()
        _=np.histogram(bkgVals,bins=binDict[var] , density=True)
         
        countsBkg, edges=_[0],_[1]
        countsBkg=countsBkg/(sum(countsBkg)+1e-9)
        ymax=max(countsBkg)
        for ii,bi in enumerate(['b1','b2','b3','b4']):
            vals=torch.masked_select(dataset[n],jetMask[bi])
            if var=='jet_phi':
                _=np.histogram(vals,bins=binDict[var] , density=True)
            else:
                _=np.histogram(vals,bins=binDict[var],density=True)
            counts, edges=_[0],_[1]
            counts=counts/sum(counts+1e-9)
            ymax=max(ymax,max(counts))
            plt.stairs(counts, edges, fill=False,label=bi)
        ax.stairs(countsBkg, edges, fill=False,label='QCD')
        ax.text(0.6,0.2,var,transform=ax.transAxes)
        ax.set_ylim([0.0,1.2*ymax])
        ax.legend()
        fname=prefix+tag+'_'+var+'.png'
        f.savefig(fname)
        result["plot_"+var]=fname

        plt.close()
    return result

def plotAllVars(dataLoader,tag,inputVarList,bMax=-1,savePrefix='./'):
    x=[]
    for i in range(len(inputVarList)):
        l=inputVarList[i].split('_')
        v=l[0]+'_'+"_".join(l[2:])
        if v not in x:
            x.append(v)
    varLabels=x
    varMap={j:i for i,j in enumerate(varLabels)}

    dataStores=getDatsetAsTensor(dataLoader,bMax)
    
    dataset=dataStores["data"].permute(2,1,0)
    labels=dataStores['extraVars']['4bIndex']
    mask=dataStores['mask']
    
    validJets = (mask>0).permute(1,0)
    
    signalMask=torch.sum(labels,dim=1)>0
    backgroundEvtMask=torch.sum(labels,dim=1)<1.0
    
    signalJetMasks=torch.logical_and(signalMask,validJets)
    bkgJetMasks=torch.logical_and(backgroundEvtMask,validJets)
    bkgMask=(labels < 1.0).permute(1,0)
    qcdMasks=torch.logical_and(bkgMask,validJets)

    jetMask={}
    jetMask['b'] =(labels > 0.5 ).permute(1,0)
    jetMask['b1']=(labels==1 ).permute(1,0)
    jetMask['b2']=(labels==2 ).permute(1,0)
    jetMask['b3']=(labels==3 ).permute(1,0)
    jetMask['b4']=(labels==4 ).permute(1,0)

    statDict={}
    statDict['Siganl']={}
    statDict['Siganl']['NEvents']=torch.sum(signalMask).item()
    statDict['Siganl']['NValid_Jets']=torch.sum(signalJetMasks).item()
    statDict['Siganl']['Nlabel_sig']=torch.sum(torch.logical_and(signalMask,jetMask['b'])).item()
    statDict['Siganl']['Nlabel_1']=torch.sum(torch.logical_and(signalMask,jetMask['b1'])).item()
    statDict['Siganl']['Nlabel_2']=torch.sum(torch.logical_and(signalMask,jetMask['b2'])).item()
    statDict['Siganl']['Nlabel_3']=torch.sum(torch.logical_and(signalMask,jetMask['b3'])).item()
    statDict['Siganl']['Nlabel_4']=torch.sum(torch.logical_and(signalMask,jetMask['b4'])).item()
    statDict['Siganl']['Nlabel_0']=torch.sum(torch.logical_and(signalMask,qcdMasks)).item()
    
    statDict['Background']={}
    statDict['Background']['NEvents']=torch.sum(backgroundEvtMask).item()
    statDict['Background']['NValid_Jets']=torch.sum(bkgJetMasks).item()
    statDict['Background']['Nlabel_sig']=torch.sum(torch.logical_and(backgroundEvtMask,jetMask['b'])).item()
    statDict['Background']['Nlabel_1']=torch.sum(torch.logical_and(backgroundEvtMask,jetMask['b1'])).item()
    statDict['Background']['Nlabel_2']=torch.sum(torch.logical_and(backgroundEvtMask,jetMask['b2'])).item()
    statDict['Background']['Nlabel_3']=torch.sum(torch.logical_and(backgroundEvtMask,jetMask['b3'])).item()
    statDict['Background']['Nlabel_4']=torch.sum(torch.logical_and(backgroundEvtMask,jetMask['b4'])).item()
    statDict['Background']['Nlabel_0']=torch.sum(torch.logical_and(backgroundEvtMask,qcdMasks)).item()


    labels = ['NEvents', 'NValid_Jets', 'Nlabel_sig', 'Nlabel_1', 'Nlabel_2','Nlabel_3','Nlabel_4','Nlabel_0']
    signal = [ statDict['Siganl'][ky] for ky in labels]
    bkg    = [ statDict['Background'][ky] for ky in labels]
    
    x = -2.5*np.arange(len(labels))  # the label locations
    width = 1.0  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width/2, signal, width,color='b', label='Signal')
    rects2 = ax.barh(x + width/2, bkg, width,color='r', label='Background')
    
    ax.set_ylabel('Item')
    ax.set_title('Number of items')
    ax.set_yticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.semilogx()
    fname1=savePrefix+'/'+tag+'_ObjectStatstatistics.png'
    fig.savefig(fname1)
    
    rslt={"plot_itemStats":fname1}
    


    rs=plotPtEtaPhiM( dataset,signalMask, varMap={'jet_pt':0,'jet_eta':1,'jet_phi':2,'jet_mass':3} ,tag=tag, savePrefix=savePrefix )
    rslt.update(rs)
    rs=plotInputVars( dataset, qcdMasks, jetMask ,varMap,tag ,prefix=savePrefix)
    rslt.update(rs)
    return rslt
