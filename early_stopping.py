import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self,savepath,patience=3,delta=0):
        self.savepath=savepath
        self.patience=patience
        self.bestscore=None
        self.delta=delta
        self.counter=0
        self.earlystop=False
    def __call__(self,score,model):
        fscore=-score
        if self.bestscore is None:
            self.bestscore=fscore
            torch.save(model.state_dict(),self.savepath)
        elif fscore<self.bestscore+self.delta:
            self.counter+=1
            if self.counter>=self.patience:
                self.earlystop=True
        else:
            self.bestscore=fscore
            torch.save(model.state_dict(),self.savepath)
            self.counter=0