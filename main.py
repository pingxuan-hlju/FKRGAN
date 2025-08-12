import torch
from torch import nn
import numpy as np
from parameters import args
import torch.nn.functional as F
import model0527
from early_stopping import EarlyStopping
from torch.utils.data import dataset,dataloader
import random
import os
from tools4roc_pr import caculate_metrics 
# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)    
torch.backends.cudnn.deterministic = True

def train(args,model,trainLoader,validLoader,i):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)
    ES=EarlyStopping("./models/Model_CNN_fold_%d.pkl"%i)
    for e in range(args.epoch):
        model.train()
        for data,label in trainLoader:
            pred,loss=model(data.to(args.device),label.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss_total=0
            for data,label in validLoader:
                pred,loss=model(data.to(args.device),label.to(args.device))
                loss_total+=loss.item()
        print('fold:'+str(i+1)+';epoch'+str(e+1)+':'+str(loss_total))
        ES(loss_total,model)
        if ES.earlystop :
            print('stop')
            break

def test(args,model,testLoader):
    test_metric = []
    for i in range(args.kfolds):
        l_m,p_m=[],[]
        model.load_state_dict(torch.load("./models/Model_CNN_fold_%d.pkl"%i))
        model.eval()
        with torch.no_grad():
            for data, label in testLoader:
                pred,loss= model(data.to(args.device),label.to(args.device))
                l_m.append(label.float())
                p_m.append(pred.cpu().detach())
        test_metric.append(caculate_metrics(torch.cat(p_m,dim=0),torch.cat(l_m,dim=0),'softmax'))
    cv_metric=np.mean(test_metric,axis=0)
    print('mean_auc:'+str(cv_metric[0])+';mean_aupr:'+str(cv_metric[1])+";accuracy:"+str(cv_metric[2])+";f1_score:"+str(cv_metric[3])+";recall:"+str(cv_metric[4]))


if __name__ == "__main__":

    # 加载数据
    common_set=torch.load('./mdd/common_set.pkl')
    train_set=torch.load('./mdd/train_set.pkl')
    test_set=torch.load('./mdd/test_set.pkl')
    GAN_embeds_train=torch.load('./mdd/GAN_train_set.pkl')
    GAN_embeds_test=torch.load('./mdd/GAN_test_set.pkl')

    for i in range(args.kfolds):
    # 生成器特征矩阵
        gan_Mfeat,gan_Dfeat = F.normalize(GAN_embeds_train['gan_Mfeat_%d'%i], dim=1),F.normalize(GAN_embeds_train['gan_Dfeat_%d'%i],dim=1)
        model=model0527.CNN_Model(args.kernel_size).to(args.device)
        model.build_fea(train_set['mm_mdF_%d'%i].to(args.device),train_set['md_%d'%i].to(args.device),
                        common_set['dd_sem'].to(args.device),gan_Mfeat.to(args.device),gan_Dfeat.to(args.device))
        trainLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_train_%d'%i], train_set['label_train_%d'%i]), batch_size=args.batch, shuffle=True, num_workers=0)
        validLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_valid_%d'%i], train_set['label_valid_%d'%i]), batch_size=args.batch, shuffle=True, num_workers=0)

        train(args,model,trainLoader,validLoader,i)
    
    gan_Mfeat,gan_Dfeat = F.normalize(GAN_embeds_test['GAN_Mfeat_test'],dim=1),F.normalize(GAN_embeds_test['GAN_Dfeat_test'],dim=1)
    testLoader = dataloader.DataLoader(model0527.ALDataset(test_set['edge'], test_set['label']), batch_size=args.batch, shuffle=False, num_workers=0)
    test_model=model0527.CNN_Model(args.kernel_size).to(args.device)
    test_model.build_fea(test_set['mm_mdF'].to(args.device),test_set['md'].to(args.device),
            common_set['dd_sem'].to(args.device),gan_Mfeat.to(args.device),gan_Dfeat.to(args.device))
    test(args,test_model,testLoader)