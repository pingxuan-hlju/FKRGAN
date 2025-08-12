import os
import torch
import torch.nn as nn
import numpy as np
from parameters import args
from sklearn import metrics


def caculate_metrics(p_m,l_m,act_type='softmax'):
    if act_type=='softmax':
        fl_p=torch.softmax(p_m,dim=1)[:,1].numpy()
        ol_p=torch.argmax(p_m,dim=1).numpy()
    elif act_type=='sigmoid':
        fl_p=p_m.numpy()
        ol_p=(p_m>0.5).long().numpy()
    l_m=l_m.numpy()
    p,r,_= metrics.precision_recall_curve(l_m,fl_p)
    metric_result=[metrics.roc_auc_score(l_m,fl_p),metrics.auc(r, p),metrics.accuracy_score(l_m,ol_p),
                                    metrics.f1_score(l_m,ol_p),metrics.recall_score(l_m,ol_p)]
    print("auc:"+str(metric_result[0])+";aupr:"+str(metric_result[1])+";accuracy:"+str(metric_result[2])+";f1_score:"+str(metric_result[3])+";recall:"+str(metric_result[4]))
    return metric_result

