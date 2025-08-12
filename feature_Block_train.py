import torch
from parameters import args
import numpy as np
import model0527
from early_stopping import EarlyStopping
from torch.utils.data import dataset,dataloader
import random
import os
# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)    
torch.backends.cudnn.deterministic = True

def normalize_adj(adj):
    degree = torch.sum(adj, dim=1)           # 度向量 [N]
    d_inv_sqrt = torch.pow(degree, -0.5)     # D^{-1/2}
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0  # 处理除以0
    D_inv_sqrt = torch.diag(d_inv_sqrt)      # 变成对角矩阵
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    return norm_adj

def feature_Block_train(model,feature,adj_hat,args,trainLoader,validLoader,key,cross):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    ES=EarlyStopping('./models/feature_Block_%s_fold_%d.pkl'%(key,cross))
    for epoch in range(args.epochs):
        model.train()
        for data,label in trainLoader:
            x1,x2,label=data[:,0].long().to(args.device),data[:,1].long().to(args.device),label.long().to(args.device)
            # 分批更新
            emb_encodeer, loss = model(feature,adj_hat,x1,x2,key)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            loss_total=0
            for data,label in validLoader:
                x1,x2,label=data[:,0].long().to(args.device),data[:,1].long().to(args.device),label.long().to(args.device)
                _, loss = model(feature,adj_hat,x1,x2,key)
                loss_total+=loss.item()
            print('fold:'+str(cross+1)+';epoch'+str(epoch+1)+':'+str(loss_total))
            ES(loss_total,model)
            if ES.earlystop:
                print('stop')
                break

# 
def dataset_save(common_set,train_set,test_set):
    FBtrain_set,FBtest_set = {},{}
    # 同质图 mm_rfam
    rfam_emb = common_set['mm_rfam'].to(args.device)
    rfam_adj = common_set['mm_rfam_adj'].to(args.device)
    HoG4rfam_hat = normalize_adj(rfam_adj.fill_diagonal_(1))
    # 同质图 mm_cluster
    cluster_emb = common_set['mm_cluster'].to(args.device)
    HoG4cluster_hat = normalize_adj(cluster_emb)
    
    # 同质图 mm_targets
    targets_emb = common_set['mm_targets'].to(args.device)
    HoG4targets = common_set['mm_targets_adj'].to(args.device)
    HoG4targets_hat = normalize_adj(HoG4targets)


    HoG4rfam_test_emb_ls,HoG4cluster_test_emb_ls,HoG4targets_test_emb_ls = [],[],[]
    for i in range(args.kfolds):
        # 异质图
        miRNA_feas=torch.cat([train_set['mm_mdF_%d'%i],train_set['md_%d'%i]],dim=1)
        dis_feas = torch.cat([train_set['md_%d'%i].T,common_set['dd_sem']],dim=1)
        HeG_embed = torch.cat([miRNA_feas,dis_feas],dim=0).to(args.device)
        HeG_hat = normalize_adj(HeG_embed)
        HeG_model = model0527.feature_Block(HeG_embed.shape[1],1024,128).to(args.device)
        HeG_model.load_state_dict(torch.load('./models/feature_Block_HeG4embed_fold_%d.pkl'%i))
        
        for param in HeG_model.parameters():
            param.requires_grad = False
        # 这里的索引是用来重构损失的，现在需要的是得到特征矩阵
        left, right =  torch.arange(0, 32).long().to(args.device), torch.arange(0, 32).long().to(args.device)
        HeG_model.eval()
        with torch.no_grad():
            emb, _= HeG_model(HeG_embed,HeG_hat,left, right,"HeG4embed")
            FBtrain_set['HeG_emb_train_%d'%i]= emb.cpu()
            
        # 同质图 mm_rfam
        HoG4rfam_model = model0527.feature_Block(rfam_emb.shape[1],1024,128).to(args.device)
        HoG4rfam_model.load_state_dict(torch.load('./models/feature_Block_miRNA_rfam4embed_fold_%d.pkl'%i))
        for param in HoG4rfam_model.parameters():
            param.requires_grad = False
        left, right =  torch.arange(0, 32).long().to(args.device), torch.arange(0, 32).long().to(args.device)
        HoG4rfam_model.eval()
        with torch.no_grad():
            emb, _= HoG4rfam_model(rfam_emb,HoG4rfam_hat,left, right,"miRNA_rfam4embed")
            HoG4rfam_test_emb_ls.append(emb)
            FBtrain_set['HoG4rfam_emb_train_%d'%i]= emb.cpu()
        
        # 同质图 mm_cluster
        HoG4cluster_model = model0527.feature_Block(cluster_emb.shape[1],1024,128).to(args.device)
        HoG4cluster_model.load_state_dict(torch.load('./models/feature_Block_miRNA_cluster4embed_fold_%d.pkl'%i))
        for param in HoG4cluster_model.parameters():
            param.requires_grad = False
        left, right =  torch.arange(0, 32).long().to(args.device), torch.arange(0, 32).long().to(args.device)
        HoG4cluster_model.eval()
        with torch.no_grad():
            emb, _= HoG4cluster_model(cluster_emb,HoG4cluster_hat,left, right,"miRNA_cluster4embed")
            HoG4cluster_test_emb_ls.append(emb)
            FBtrain_set['HoG4cluster_emb_train_%d'%i]= emb.cpu()
        
        # 同质图 mm_targets
        HoG4targets_model = model0527.feature_Block(targets_emb.shape[1],1024,128).to(args.device)
        HoG4targets_model.load_state_dict(torch.load('./models/feature_Block_miRNA_targets4embed_fold_%d.pkl'%i))
        for param in HoG4targets_model.parameters():
            param.requires_grad = False
        left, right =  torch.arange(0, 32).long().to(args.device), torch.arange(0, 32).long().to(args.device)
        HoG4targets_model.eval()
        with torch.no_grad():
            emb, _= HoG4targets_model(targets_emb,HoG4targets_hat,left, right,"miRNA_targets4embed")
            HoG4targets_test_emb_ls.append(emb)
            FBtrain_set['HoG4targets_emb_train_%d'%i]= emb.cpu()
    
    torch.save(FBtrain_set,'./mdd/FBtrain_set.pkl')
    print('FBtrain_set saved')

    HoG4rfam_emb = torch.stack(HoG4rfam_test_emb_ls).mean(0)
    HoG4cluster_emb = torch.stack(HoG4cluster_test_emb_ls).mean(0)
    HoG4targets_emb = torch.stack(HoG4targets_test_emb_ls).mean(0)
    FBtest_set['HoG4rfam_emb_test'] = HoG4rfam_emb.cpu()
    FBtest_set['HoG4cluster_emb_test'] = HoG4cluster_emb.cpu()
    FBtest_set['HoG4targets_emb_test'] = HoG4targets_emb.cpu()

    # 保存训练集所生成的特征
    test_emb_ls = []
    miRNA_feas=torch.cat([test_set['mm_mdF'],test_set['md']],dim=1)
    dis_feas = torch.cat([test_set['md'].T,common_set['dd_sem']],dim=1)
    HeG_embed = torch.cat([miRNA_feas,dis_feas],dim=0).to(args.device)
    HeG_hat = normalize_adj(HeG_embed)
    HeG_model1 = model0527.feature_Block(HeG_embed.shape[1],1024,128).to(args.device)
    for i in range(args.kfolds):
        HeG_model1.load_state_dict(torch.load('./models/feature_Block_HeG4embed_fold_%d.pkl'%i))
        for param in HeG_model1.parameters():
            param.requires_grad = False
        # 这里的索引是用来重构损失的，现在需要的是得到特征矩阵
        left, right =  torch.arange(0, 32).long().to(args.device), torch.arange(0, 32).long().to(args.device)
        HeG_model1.eval()
        with torch.no_grad():
            emb, _= HeG_model1(HeG_embed,HeG_hat,left, right,"HeG4embed")
            test_emb_ls.append(emb)
    avg_emb = torch.stack(test_emb_ls).mean(0)
    FBtest_set['HeG_emb_test'] = avg_emb.cpu()
    torch.save(FBtest_set,'./mdd/FBtest_set.pkl')
    print('FBtest_set saved')

    

if __name__ == "__main__":
    common_set=torch.load('./mdd/common_set.pkl')
    train_set=torch.load('./mdd/train_set.pkl')
    test_set=torch.load('./mdd/test_set.pkl')
    
    for i in range(args.kfolds):
        # 异质图 -特征矩阵
        miRNA_feas=torch.cat([train_set['mm_mdF_%d'%i],train_set['md_%d'%i]],dim=1)
        dis_feas = torch.cat([train_set['md_%d'%i].T,common_set['dd_sem']],dim=1)
        feature = torch.cat([miRNA_feas,dis_feas],dim=0).to(args.device)
        # 邻接矩阵
        HeG_hat = normalize_adj(feature)
        model = model0527.feature_Block(feature.shape[1],1024,128).to(args.device)
        trainLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_train_%d'%i], train_set['label_train_%d'%i]), batch_size=8192, shuffle=True, num_workers=0)
        validLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_valid_%d'%i], train_set['label_valid_%d'%i]), batch_size=10240, shuffle=True, num_workers=0)
        get_HeG4embed = feature_Block_train(model,feature,HeG_hat,args,trainLoader,validLoader,"HeG4embed",i)
    
    for i in range(args.kfolds):    
        # 同质图 mm_rfam
        rfam_emb = common_set['mm_rfam'].to(args.device)
        rfam_adj = common_set['mm_rfam_adj'].to(args.device)
        HoG4rfam_hat = normalize_adj(rfam_adj.fill_diagonal_(1))
        net1 = model0527.feature_Block(rfam_emb.shape[1],1024,128).to(args.device)
        trainLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_train_%d'%i], train_set['label_train_%d'%i]), batch_size=8192, shuffle=True, num_workers=0)
        validLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_valid_%d'%i], train_set['label_valid_%d'%i]), batch_size=10240, shuffle=True, num_workers=0)
        get_HoG4rfam = feature_Block_train(net1,rfam_emb,HoG4rfam_hat,args,trainLoader,validLoader,"miRNA_rfam4embed",i)
    
    for i in range(args.kfolds):
        # 同质图 mm_cluster
        cluster_emb = common_set['mm_cluster'].to(args.device)
        HoG4cluster_hat = normalize_adj(cluster_emb)
        net2 = model0527.feature_Block(cluster_emb.shape[1],1024,128).to(args.device)
        trainLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_train_%d'%i], train_set['label_train_%d'%i]), batch_size=8192, shuffle=True, num_workers=0)
        validLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_valid_%d'%i], train_set['label_valid_%d'%i]), batch_size=10240, shuffle=True, num_workers=0)
        get_HoG4cluster = feature_Block_train(net2,cluster_emb,HoG4cluster_hat,args,trainLoader,validLoader,"miRNA_cluster4embed",i)
    
    for i in range(args.kfolds):
        # 同质图 mm_targets
        targets_emb = common_set['mm_targets'].to(args.device)
        HoG4targets = common_set['mm_targets_adj'].to(args.device)
        HoG4targets_hat = normalize_adj(HoG4targets)
        net3 = model0527.feature_Block(targets_emb.shape[1],1024,128).to(args.device)
        trainLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_train_%d'%i], train_set['label_train_%d'%i]), batch_size=8192, shuffle=True, num_workers=0)
        validLoader = dataloader.DataLoader(model0527.ALDataset(train_set['edge_valid_%d'%i], train_set['label_valid_%d'%i]), batch_size=10240, shuffle=True, num_workers=0)   
        get_HoG4cluster = feature_Block_train(net3,targets_emb,HoG4targets_hat,args,trainLoader,validLoader,"miRNA_targets4embed",i)    
    
    dataset_save(common_set,train_set,test_set)