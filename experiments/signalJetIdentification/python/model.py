import torch
import torch.utils.data as data
from TransformerModel import *

class trippleHDataset(data.Dataset):
    def __init__(self, features, labels,masks=None,extraVals=None, evaluation=True):
        """
        Inputs:
            features - Tensor of shape [num_evts, evt_dim]. Represents the high-level features.
            labels - Tensor of shape [num_evts], containing the class labels for the events
            train - If True nothing will happen
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.extraVals = extraVals
        self.evaluation = evaluation
        self.mask = masks
        if masks==None:
            print("Initializing with default mask ")
            self.mask=torch.ones(self.labels.shape)
        if extraVals is None:
            self.extraVals={}

        # Tensors with indices of the images per class
        self.num_labels = labels.max()+1

    def __len__(self):
        return self.features.shape[0]

    def setEval(self,flag=True):
        self.evaluation=flag   

    def __getitem__(self, idx):
        # We return the indices of the event for visualization purpose."Label" is the class
        if self.evaluation:
            return self.features[idx],self.mask[idx],self.labels[idx],{tag:self.extraVals[tag][idx] for tag in self.extraVals}
            
        return self.features[idx], self.mask[idx], self.labels[idx]


class trippleHNonResonatModel(TransformerPredictor):
    def __init__(self,inputVarList=[],remark='',**kwargs):
        
        #print(kwargs)
        print("  Model Dims                 : ",kwargs['model_dim'])
        print("  Number of Heads in MHA     : ",kwargs['num_heads'])
        print("  Heads dim in MHA           : ",kwargs['model_dim']/kwargs['num_heads'])
        print("  Total Encoder layers  : ",kwargs['num_layers'])
        self.inputVars=inputVarList
        self.remark=remark
        super(trippleHNonResonatModel,self).__init__(**kwargs)
        
        # Output softmax
#        self.descriminator_net = torch.nn.Softmax(dim=1)

        self.save_hyperparameters()

    def _calculate_loss(self, batch, mode="train"):
        features, mask, labels = batch
       
        # Perform prediction and calculate loss and accuracy
        preds = self.forward(features,mask=mask, add_positional_encoding=False)
        loss = F.cross_entropy(preds.view(-1,preds.size(-1))[mask.view(-1) > 0.5 ], labels.view(-1)[mask.view(-1)>0.5])
        acc = (preds.argmax(dim=-1) == labels.squeeze()).float().mean()        
        
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True)

        return loss, acc
    
    def forward(self, x, mask=None, add_positional_encoding=False):
        if mask != None:
            mask=torch.bmm(torch.unsqueeze(mask,-1),torch.unsqueeze(mask,-2))
        x= super(trippleHNonResonatModel,self).forward(x,mask=mask, add_positional_encoding=add_positional_encoding)
#        x= self.descriminator_net(x)
        return x
        
        
    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


