from sklearn import metrics
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getEvaluationResults(dataLoader,hhhVsQCD_model,bMax=-1):
    evalResult={}
    bCount=0
    if bMax < 0 :
        bMax=len(dataLoader)
    with torch.no_grad():
        for batch in iter(dataLoader):
            if bCount%50==0:
                print("\t Processing batch ",bCount," [ max : ",bMax," ] ")
            if bCount>=bMax:
                break
            x,mask,label,extraVars=batch
            x=x.to(hhhVsQCD_model.device)
            maskCu=mask.to(hhhVsQCD_model.device)
            rslt=hhhVsQCD_model(x,mask=maskCu)
            rslt=rslt.to('cpu')

            mask=mask>0.5
            if 'yPred' not in evalResult:
                evalResult={}
                evalResult['yPred']=rslt
                evalResult['label']=label
                evalResult['vldMask']=mask
                for ky in extraVars:
                    evalResult[ky]=extraVars[ky]
                    
            else:
                evalResult['yPred']=torch.concat([evalResult['yPred'],rslt])
                evalResult['label']=torch.concat([evalResult['label'],label])
                evalResult['vldMask']=torch.concat([evalResult['vldMask'],mask])
                for ky in extraVars:
                    evalResult[ky]=torch.concat([evalResult[ky],extraVars[ky]])
            bCount+=1
    print("Processed ",bCount," batches out of ",len(dataLoader)," available batches | batch size = ",dataLoader.batch_size)
    return evalResult

def getAccuracyAndLoss(dataLoaders):
    print("Testing the dataloaders for accuracy and loss ! ")
    result={}
    for tag in dataLoaders:
        print("tag : ",tag," nBatches : ",len(dataLoaders[tag]),
              " | n events = ",len(dataLoaders[tag])*dataLoaders[tag].batch_size)
        result[tag]=tester.test(hhhVsQCD_model,dataLoaders[tag])
    return result

def saveEval_Y0Y1plots(evalResult,hhhTo4bMask,tag='',savePrefix='./'):
    y0PredSig=evalResult['yPred'][evalResult['vldMask']][:,1].detach().numpy()
    y1PredSig=evalResult['yPred'][evalResult['vldMask']][:,0].detach().numpy()
    
    xEdges=np.arange(0.0,1.0,0.01)
    yEdges=np.arange(0.0,1.0,0.01)
    
    f,ax=plt.subplots(figsize=(12,5),nrows=1,ncols=2)
    
    if(len(y0PredSig)>0):
        counts, xedges, yedges, im=ax[0].hist2d(sigmoid(y0PredSig),sigmoid(y1PredSig),
                       bins=[xEdges,yEdges],
                       density=True,
                       norm=mpl.colors.LogNorm(),
                       cmap=plt.cm.plasma);
        f.colorbar(im, ax=ax[0])
    ax[0].text(0.6,0.8,tag,fontweight='bold',color='white')
    ax[0].set_facecolor("black")
    
    fname=savePrefix+'/'+tag+'Y0Y1_2D.png'
    f.savefig(fname,bbox_inches='tight')
    
    f,ax=plt.subplots(figsize=(12,5),nrows=1,ncols=2)
    _=ax[0].hist(sigmoid(y0PredSig),bins=50,label='Signal',histtype='step',density=True,color='b')
    ax[0].set_xlabel('y0')
    ax[0].legend(loc=0)
    _=ax[1].hist(sigmoid(y1PredSig),bins=50,label='Signal',histtype='step',density=True,color='b')
    ax[1].legend(loc=2)
    ax[1].set_xlabel('y1')    

    fnamePrj=savePrefix+'/'+tag+'Y0_Y1_projection.png'
    f.savefig(fnamePrj,bbox_inches='tight')

    my_cmap = plt.cm.plasma
    my_cmap = plt.cm.plasma
    my_cmap.set_under('k',0)
    
    f,ax=plt.subplots(figsize=(10,6),nrows=2,ncols=2)
    ax=np.ndarray.flatten(ax)
    rw=0
    for ky in ['4bMask_1','4bMask_2','4bMask_3','4bMask_4']:
        y0PredSig=evalResult['yPred'][hhhTo4bMask[ky]][:,1].detach().numpy()
        y1PredSig=evalResult['yPred'][hhhTo4bMask[ky]][:,0].detach().numpy()
        if (len(y0PredSig) > 0):
            counts, xedges, yedges, im=ax[rw].hist2d(sigmoid(y0PredSig),sigmoid(y1PredSig),
                           bins=[xEdges,yEdges],
                         density=True,
                         norm=mpl.colors.LogNorm(),
                         cmin=0.0,
                         cmap=my_cmap);
            f.colorbar(im, ax=ax[rw])
        ax[rw].text(0.7,0.8,'Signal',fontweight='bold',color='white')
        ax[rw].text(0.7,0.70,'B'+str(rw+1),fontweight='bold',color='white')
        ax[rw].set_facecolor("black")
    
        rw+=1

    fname_SignalVsBkg_Y0_Y1_4Bs=savePrefix+'/'+tag+'_Y0_Y1_4Bs.png'
    f.savefig(fname_SignalVsBkg_Y0_Y1_4Bs,bbox_inches='tight')
    
    f,ax=plt.subplots(figsize=(10,6),nrows=2,ncols=2)
    
    ax=np.ndarray.flatten(ax)
    rw=0
    for ky in ['4bMask_1','4bMask_2','4bMask_3','4bMask_4']:
        y0PredSig=evalResult['yPred'][hhhTo4bMask[ky]][:,1].detach().numpy()
        y1PredSig=evalResult['yPred'][hhhTo4bMask[ky]][:,0].detach().numpy()
    
        if (len(y0PredSig) > 0):
            _=ax[rw].hist(sigmoid(y0PredSig),bins=50,label='Y0',histtype='step',density=True,color='red')
            _=ax[rw].hist(sigmoid(y1PredSig),bins=50,label='Y1',histtype='step',density=True,color='blue')
            ax[rw].legend(loc=2)
        
        ax[rw].text(0.7,0.80,'Signal',fontweight='bold',color='k',transform=ax[rw].transAxes)
        ax[rw].text(0.7,0.70,'B'+str(rw+1),fontweight='bold',color='k',transform=ax[rw].transAxes)
    #     ax[rw].set_facecolor("black")
    
        rw+=1
    fname_SignalVsBkg_Y0_Y1_4Bs_1d=savePrefix+'/'+tag+'_SignalVsBkg_Y0_Y1_4Bs_1d.png'
    f.savefig(fname_SignalVsBkg_Y0_Y1_4Bs_1d,bbox_inches='tight')


    return {
            'plot_2d_allJets':fname,
            'plot_1d_allJets':fnamePrj,
            'plot_2d_SVsB_Y0_Y1_4Bs':fname_SignalVsBkg_Y0_Y1_4Bs,
            'plot_2d_SVsB_Y0_Y1_4Bs_1d':fname_SignalVsBkg_Y0_Y1_4Bs_1d,
           }




def saveSignalVsBkg_Y0Y1plots(evalResult,hhhTo4bMask,tag='',savePrefix='./'):
    y0PredSig=evalResult['yPred'][hhhTo4bMask['signal']][:,1].detach().numpy()
    y1PredSig=evalResult['yPred'][hhhTo4bMask['signal']][:,0].detach().numpy()
    
    y0PredBkg=evalResult['yPred'][hhhTo4bMask['background']][:,1].detach().numpy()
    y1PredBkg=evalResult['yPred'][hhhTo4bMask['background']][:,0].detach().numpy()

    xEdges=np.arange(0.0,1.0,0.01)
    yEdges=np.arange(0.0,1.0,0.01)
    
    f,ax=plt.subplots(figsize=(12,5),nrows=1,ncols=2)
    
    if(len(y0PredBkg)>0):
        counts, xedges, yedges, im=ax[0].hist2d(sigmoid(y0PredBkg),sigmoid(y1PredBkg),
                       bins=[xEdges,yEdges],
                       density=True,
                       norm=mpl.colors.LogNorm(),
                       cmap=plt.cm.plasma);
        f.colorbar(im, ax=ax[0])
    ax[0].text(0.6,0.8,'Background',fontweight='bold',color='white')
    ax[0].set_facecolor("black")
    
    my_cmap = plt.cm.plasma
    my_cmap = plt.cm.plasma
    my_cmap.set_under('k',0)
    
    if(len(y0PredSig)>0):
        counts, xedges, yedges, im=ax[1].hist2d(sigmoid(y0PredSig),sigmoid(y1PredSig),
                       bins=[xEdges,yEdges],
        #              xedges=xEdges,yedges=yEdges,
                     density=True,
                     norm=mpl.colors.LogNorm(),
                     cmin=0.0,
                     cmap=my_cmap);
        f.colorbar(im, ax=ax[1])
    ax[1].text(0.7,0.8,'Signal',fontweight='bold',color='white')
    ax[1].set_facecolor("black")
    
    fname=savePrefix+'/'+tag+'_SignalVsBkg_Y0Y1_2D.png'
    f.savefig(fname,bbox_inches='tight')
    
    f,ax=plt.subplots(figsize=(12,5),nrows=1,ncols=2)
    _=ax[0].hist(sigmoid(y0PredSig),bins=50,label='Signal',histtype='step',density=True,color='b')
    _=ax[0].hist(sigmoid(y0PredBkg),bins=50,label='Background',histtype='step',density=True,color='r')
    ax[0].set_xlabel('y0')
    ax[0].legend(loc=0)
    _=ax[1].hist(sigmoid(y1PredSig),bins=50,label='Signal',histtype='step',density=True,color='b')
    _=ax[1].hist(sigmoid(y1PredBkg),bins=50,label='Background',histtype='step',density=True,color='r')
    ax[1].legend(loc=2)
    ax[1].set_xlabel('y1')    

    fnamePrj=savePrefix+'/'+tag+'_SignalVsBkg_Y0_Y1_projection.png'
    f.savefig(fnamePrj,bbox_inches='tight')

    f,ax=plt.subplots(figsize=(10,6),nrows=2,ncols=2)
    ax=np.ndarray.flatten(ax)
    rw=0
    for ky in ['4bMask_1','4bMask_2','4bMask_3','4bMask_4']:
        y0PredSig=evalResult['yPred'][hhhTo4bMask[ky]][:,1].detach().numpy()
        y1PredSig=evalResult['yPred'][hhhTo4bMask[ky]][:,0].detach().numpy()
        if (len(y0PredSig) > 0):
            counts, xedges, yedges, im=ax[rw].hist2d(sigmoid(y0PredSig),sigmoid(y1PredSig),
                           bins=[xEdges,yEdges],
                         density=True,
                         norm=mpl.colors.LogNorm(),
                         cmin=0.0,
                         cmap=my_cmap);
            f.colorbar(im, ax=ax[rw])
        ax[rw].text(0.7,0.8,'Signal',fontweight='bold',color='white')
        ax[rw].text(0.7,0.70,'B'+str(rw+1),fontweight='bold',color='white')
        ax[rw].set_facecolor("black")
    
        rw+=1

    fname_SignalVsBkg_Y0_Y1_4Bs=savePrefix+'/'+tag+'_SignalVsBkg_Y0_Y1_4Bs.png'
    f.savefig(fname_SignalVsBkg_Y0_Y1_4Bs,bbox_inches='tight')
    
    f,ax=plt.subplots(figsize=(10,6),nrows=2,ncols=2)
    ax=np.ndarray.flatten(ax)
    rw=0
    for ky in ['4bMask_1','4bMask_2','4bMask_3','4bMask_4']:
        y0PredSig=evalResult['yPred'][hhhTo4bMask[ky]][:,1].detach().numpy()
        y1PredSig=evalResult['yPred'][hhhTo4bMask[ky]][:,0].detach().numpy()
    
        _=ax[rw].hist(sigmoid(y0PredSig),bins=50,label='Y0',histtype='step',density=True,color='red')
        _=ax[rw].hist(sigmoid(y1PredSig),bins=50,label='Y1',histtype='step',density=True,color='blue')
        
        ax[rw].text(0.7,0.80,'Signal',fontweight='bold',color='k',transform=ax[rw].transAxes)
        ax[rw].text(0.7,0.70,'B'+str(rw+1),fontweight='bold',color='k',transform=ax[rw].transAxes)
        ax[rw].legend(loc=2)
    #     ax[rw].set_facecolor("black")
    
        rw+=1
    fname_SignalVsBkg_Y0_Y1_4Bs_1d=savePrefix+'/'+tag+'_SignalVsBkg_Y0_Y1_4Bs_1d.png'
    f.savefig(fname_SignalVsBkg_Y0_Y1_4Bs_1d,bbox_inches='tight')


    return {
            'plot_2d_allJets':fname,
            'plot_1d_allJets':fnamePrj,
            'plot_2d_SVsB_Y0_Y1_4Bs':fname_SignalVsBkg_Y0_Y1_4Bs,
            'plot_2d_SVsB_Y0_Y1_4Bs_1d':fname_SignalVsBkg_Y0_Y1_4Bs_1d,
           }

def saveSignalVsBkg_EventLevelPlots(evalResult,savePrefix='./'):

    signalEventMask=torch.sum(evalResult['label'],dim=-1)==4
    bkgEventMask=torch.sum(evalResult['label'],dim=-1)<4
    
    mask=evalResult['vldMask']
    m2=torch.logical_not(mask)
    yPred=evalResult[tag]['yPred'].clone()
    yPred[m2]=-1e9
    preds=torch.argmax(yPred,dim=-1)

    sumJets=torch.sum(preds,dim=1).detach().numpy()
    
    plt.hist(sumJets[signalEventMask],bins=np.arange(-0.5,10.5,1.0),histtype='step',color='b',label='Signal',density=True)
    plt.hist(sumJets[bkgEventMask],bins=np.arange(-0.5,10.5,1.0),histtype='step',color='r',label='Background',density=True)
    plt.xlabel('nJets tagged signal')
    plt.semilogy()
    fname_nJetsPositivePerEvt=savePrefix+'/'+tag+'_nJets_tagged_signalPerEvent.png'
    f.savefig(fname_nJetsPositivePerEvt,bbox_inches='tight')

    return {
            'plot_nJets_tagged_signalPerEvent': fname_nJetsPositivePerEvt,
           }

def saveBinaryClassifier_PerformanceEvalPlots(model,evalResult,hhhTo4bMask,tag='',savePrefix='./'):
    
    testX=evalResult['yPred'][evalResult['vldMask']]
    pred_y = model(testX)
    yPrTest=pred_y.view(-1).detach().numpy()
    
    plt.figure()
    _=plt.hist(yPrTest,bins=40,histtype='step',density=True,log=True)
    
    plt.legend(loc=9)
    plt.title('Score')
    fname_overTrainingChk=savePrefix+'/'+tag+'_binaryClassification_scores.png'
    plt.savefig(fname_overTrainingChk,bbox_inches='tight')
    plt.close()
    
    
    plt.figure()
    _=plt.hist(yPrTest,bins=40,histtype='step',density=True,cumulative=-1)
    plt.xlabel('score')
    plt.ylabel('effic')
    plt.title('Efficiency')

    fname_binrayCls_ess=savePrefix+'/'+tag+'_binCls_eff.png'
    plt.savefig(fname_binrayCls_ess,bbox_inches='tight')
    plt.close()

    f,ax=plt.subplots(figsize=(10,6),nrows=2,ncols=2)
    ax=np.ndarray.flatten(ax)
    rw=0
    for ky in ['4bMask_1','4bMask_2','4bMask_3','4bMask_4']:
        test_x=evalResult['yPred'][hhhTo4bMask[ky]]
        pred_y = model(test_x)
        yPrTest=pred_y.view(-1).detach().numpy()
    
        _=ax[rw].hist(yPrTest,bins=40,histtype='step',density=True)
        
        ax[rw].text(0.5,0.80,'Signal',fontweight='bold',color='k',transform=ax[rw].transAxes)
        ax[rw].text(0.5,0.70,'B'+str(rw+1),fontweight='bold',color='k',transform=ax[rw].transAxes)
        ax[rw].set_xlabel("BinaryClass. Score B"+str(rw+1))
        ax[rw].legend(loc=2)
    #     ax[rw].set_facecolor("black")
    
        rw+=1
    fname_binaryClassification_scores_4bs=savePrefix+'/'+tag+'_binaryClassification_scores_4bs.png'
    f.savefig(fname_binaryClassification_scores_4bs,bbox_inches='tight')


    return {
            'plot_binCls_scores': fname_overTrainingChk,
            'plot_binCls_scores_4bs': fname_binaryClassification_scores_4bs,
            'plot_binCls_eff': fname_binrayCls_ess
        }



def saveBinaryClassifier_PerformancePlots(model,trainX,trainY,testX,testY,savePrefix='./'):
    pred_y = model(trainX)
    
    m1= trainY==1
    m2= trainY<1
    
    
    plt.figure()
    yPr=pred_y.detach().numpy()
    
    _=plt.hist(yPr[m1],bins=40,histtype='step',log=True,density=True,label='Signal,Train')
    _=plt.hist(yPr[m2],bins=40,histtype='step',log=True,density=True,label='Background,Train')
    
    pred_y = model(testX)
    m1= testY==1 ;    m2= testY<1
    
    yPrTest=pred_y.detach().numpy()
    print(yPrTest.shape)
    
    _=plt.hist(yPrTest[m1],bins=40,histtype='step',density=True,log=True,label='Signal,Test')
    _=plt.hist(yPrTest[m2],bins=40,histtype='step',density=True,log=True,label='Background,Test')
    
    plt.legend(loc=9)
    plt.title('Overtraining Check')
    fname_overTrainingChk=savePrefix+'/plot_binaryClassificatin_overtraining_check.png'
    plt.savefig(fname_overTrainingChk,bbox_inches='tight')
    plt.close()
    
    plt.figure()
    pred_y = model(testX).detach().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(testY.detach().numpy(), pred_y, pos_label=1)
    rocAuc=metrics.roc_auc_score(testY.detach().numpy(), pred_y)
    
    idx=np.argmin(abs(tpr-0.9))
    eff=np.round(tpr[idx],3)
    fkRt=np.round(fpr[idx],3)
    thr=np.round(thresholds[idx],3)
    rocAuc=np.round(rocAuc,3)
    
    plt.plot(tpr,1-fpr)
    plt.text(0.2,0.6,"AUC : "+str(rocAuc))
    plt.text(0.2,0.5,"TPR = "+str(eff)+" ==> thresholds= "+str(thr))
    plt.text(0.2,0.4," TPR = "+str(eff)+" @ "+ ' FPR = '+str(fkRt))
    plt.xlabel('TPR')
    plt.ylabel('1-FPR')
    plt.title('ROC Curve')

    fname_binrayCls_roc=savePrefix+'/plot_binCls_roc.png'
    plt.savefig(fname_binrayCls_roc,bbox_inches='tight')
    plt.close()

    return {
            'plot_binCls_overTrainingCheck': fname_overTrainingChk,
            'plot_roc' : fname_binrayCls_roc,
            'roc' : rocAuc,
            'eff' : eff,
            'fakeRate' : fkRt,
            'threshold' : thr
        }

def saveSignalVsBkg_binaryEventLevelEvalPlots(binaryModel,evalResult,thr=0.5,tag='',savePrefix='./'):

    signalEventMask=torch.sum(evalResult['label'],dim=-1)>0
    bkgEventMask=torch.sum(evalResult['label'],dim=-1)<1
    
    pred_y = binaryModel(evalResult['yPred'])
    mask=evalResult['vldMask']
    m2=torch.logical_not(mask)
    pred_y[m2]=thr-10
    preds=pred_y>thr

    sumJets=torch.sum(preds,dim=1).detach().numpy()
    plt.figure()
    if(torch.sum(signalEventMask) > 0):
        plt.hist(sumJets[signalEventMask],bins=np.arange(-0.5,10.5,1.0),
                 histtype='step',color='b',label='Signal',density=True)
    if(torch.sum(bkgEventMask) > 0):
        plt.hist(sumJets[bkgEventMask],bins=np.arange(-0.5,10.5,1.0),
             histtype='step',color='r',label='Background',density=True)
    plt.xlabel('nJets tagged signal / Event')
    plt.legend(loc=1)

    fname_nJetsPositivePerEvt=savePrefix+'/'+tag+'_nJets_tagged_signalPerEvent_BinaryCls.png'
    plt.text(2.0,0.8,"THR = "+str(thr),fontweight='bold',color='k')
    plt.savefig(fname_nJetsPositivePerEvt,bbox_inches='tight')
    plt.close()
    plt.title(tag)
    return {
            'plot_nJets_tagged_signalPerEvent': fname_nJetsPositivePerEvt,
           }





def saveSignalVsBkg_binaryEventLevelPlots(binaryModel,evalResult,thr=0.5,tag='',savePrefix='./'):

    signalEventMask=torch.sum(evalResult['label'],dim=-1)>0
    bkgEventMask=torch.sum(evalResult['label'],dim=-1)<1
    
    pred_y = binaryModel(evalResult['yPred'])
    mask=evalResult['vldMask']
    m2=torch.logical_not(mask)
    pred_y[m2]=thr-10
    preds=pred_y>thr

    sumJets=torch.sum(preds,dim=1).detach().numpy()
    plt.figure()
    plt.hist(sumJets[signalEventMask],bins=np.arange(-0.5,10.5,1.0),
             histtype='step',color='b',label='Signal',density=True)
    plt.hist(sumJets[bkgEventMask],bins=np.arange(-0.5,10.5,1.0),
             histtype='step',color='r',label='Background',density=True)
    plt.xlabel('nJets tagged signal / Event')
    plt.legend(loc=1)

    fname_nJetsPositivePerEvt=savePrefix+'/'+tag+'_nJets_tagged_signalPerEvent_BinaryCls.png'
    plt.savefig(fname_nJetsPositivePerEvt,bbox_inches='tight')
    plt.close()
    plt.title(tag)
    return {
            'plot_nJets_tagged_signalPerEvent': fname_nJetsPositivePerEvt,
           }



