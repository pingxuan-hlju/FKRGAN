import os
import torch
import numpy as np
from parameters import args
from sklearn.model_selection import KFold
import random
import itertools
# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)    
torch.backends.cudnn.deterministic = True

class dataloader(object):
    def __init__(self, params):
        self.params = params
        self.mm_seq = np.load(self.params.miRNA_simi_path)
        self.dd_sem = np.load(self.params.disease_simi_path)
        self.md = np.load(self.params.miRNA_disease_adj_path)
        self.miRNA_rfam_mat = np.load(self.params.miRNA_rfam_path)
        self.miRNA_cluster = np.load(self.params.miRNA_cluster_path)
        self.miRNA_targets = np.load(self.params.miRNA_targets_path)
        self.miRNA_targets_adj = np.load(self.params.miRNA_targets_adj_path)
        self.split_dataset()

    def load_data(self):
        common_set={}
        common_set['md']= torch.tensor(self.md).long()
        common_set['mm_seq']= torch.tensor(self.mm_seq).float()
        common_set['dd_sem']= torch.tensor(self.dd_sem).float()
        common_set['mm_rfam']= torch.tensor(self.miRNA_rfam_mat).float()
        common_set['mm_cluster']= torch.tensor(self.miRNA_cluster).float()
        common_set['mm_targets']= torch.tensor(self.miRNA_targets).float()
        common_set['mm_targets_adj']= torch.tensor(self.miRNA_targets_adj).float()
        common_set['mm_rfam_adj']= self.get_rfam_adj(self.miRNA_rfam_mat)
        torch.save(common_set,'./mdd/common_set.pkl')
        print('common_set saved')
        return common_set
    
    def get_rfam_adj(self,feas):
        emb = torch.tensor(feas).float()
        # 初始化邻接矩阵
        N = emb.shape[0]
        adj = torch.zeros(N,N)
        # 构建邻接：如果两个节点家族向量相同（且非零），连边
        # 找出非全零的节点索引
        nonzero_idx = [i for i in range(N) if not torch.all(emb[i] == 0)]

        for i in nonzero_idx:
            for j in nonzero_idx:
                if i < j and torch.all(emb[i] == emb[j]):
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj
    
    def GIP_sim(self,matrix):
        matrix=matrix.float()
        fz=(matrix*matrix).sum(dim=1,keepdims=True)+(matrix*matrix).sum(dim=1,keepdims=True).T-2*matrix@matrix.T
        fm=1/torch.diag(matrix@matrix.T).mean()
        return torch.exp(-1*fz*fm)
    
    def Functional_sim(self,ass,sim,device,batch=1): #(a,b) #(b,b)
        s1=ass.shape[0]      #a
        ass=ass.to(device)
        sim=sim.to(device)
        sim_m=torch.zeros(s1,s1).to(device)  #(a,a)
        iter_comb=torch.tensor(list(itertools.combinations(range(s1),2))).long().to(device)
        for i in range(iter_comb.shape[0]//batch):
            idx1,idx2=iter_comb[i*batch:(i+1)*batch,0],iter_comb[i*batch:(i+1)*batch,1]
            m1=ass[idx1,:]
            m2=ass[idx2,:]
            sim1=m1[:,:,None]*sim*m2[:,None,:]  # (batch,b,b)
            sim_m[idx1,idx2]=(sim1.max(dim=1)[0].sum(dim=-1)+sim1.max(dim=2)[0].sum(dim=-1))/(m1.sum(dim=-1)+m2.sum(dim=-1))
        if iter_comb.shape[0]%batch!=0:
            idx1,idx2=iter_comb[(i+1)*batch:,0],iter_comb[(i+1)*batch:,1]
            m1=ass[idx1,:]
            m2=ass[idx2,:]
            sim1=m1[:,:,None]*sim*m2[:,None,:]  # (batch,b,b)
            sim_m[idx1,idx2]=(sim1.max(dim=1)[0].sum(dim=-1)+sim1.max(dim=2)[0].sum(dim=-1))/(m1.sum(dim=-1)+m2.sum(dim=-1))
        sim_m=torch.where(torch.isinf(sim_m),torch.zeros_like(sim_m),sim_m)
        sim_m=torch.where(torch.isnan(sim_m),torch.zeros_like(sim_m),sim_m)
        return (sim_m+sim_m.T+torch.eye(s1).to(device)).cpu()

    def split_dataset(self):
        # 对正例划分训练集和测试集合 占比 8:2
        pos_x,pos_y=np.where(self.md==1)
        pos_xy=np.concatenate([pos_x[:,None],pos_y[:,None]],axis=1) #(23337, 2)
        pos_xy=pos_xy[np.random.permutation(pos_xy.shape[0]),:]
        train_pos_xy=pos_xy[:int(pos_xy.shape[0]*self.params.train_ratio),:]
        test_pos_xy=pos_xy[int(pos_xy.shape[0]*self.params.train_ratio):,:]
        
        # 对负例划分训练集和测试集合 占比 8:2
        neg_x,neg_y=np.where(self.md==0)
        neg_xy=np.concatenate([neg_x[:,None],neg_y[:,None]],axis=1) #(2562528, 2)
        neg_xy=neg_xy[np.random.permutation(neg_xy.shape[0]),:]
        train_neg_xy=neg_xy[:int(neg_xy.shape[0]*self.params.train_ratio),:]
        test_neg_xy=neg_xy[int(neg_xy.shape[0]*self.params.train_ratio):,:]

        # 获取训练样本的索引(index)和标签(label)
        train_xy=np.concatenate([train_pos_xy,train_neg_xy],axis=0)
        train_label=np.concatenate([np.ones(train_pos_xy.shape[0]),np.zeros(train_neg_xy.shape[0])],axis=0)
        train_rd=np.random.permutation(train_xy.shape[0])
        train_xy,train_label=train_xy[train_rd,:],train_label[train_rd]
        # 获取测试样本的索引(index)和标签(label)
        test_xy=np.concatenate([test_pos_xy,test_neg_xy],axis=0)
        test_label=np.concatenate([np.ones(test_pos_xy.shape[0]),np.zeros(test_neg_xy.shape[0])],axis=0)

        kf = KFold(n_splits=self.params.kfolds, shuffle=True, random_state=self.params.seed)
        # 将训练集数据再次划分为 1.训练集； 2.验证集  占比为 4 : 1
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_xy):
            train_idx.append(train_index)
            valid_idx.append(valid_index)

        train_set,test_set={},{}
        common_set = self.load_data()
        # @ mask test
        test_md=np.zeros(self.md.shape)
        test_md[train_pos_xy[:,0],train_pos_xy[:,1]]=1
        test_set['edge']=torch.tensor(test_xy).long()
        test_set['label']=torch.tensor(test_label).long()
        test_set['md']=torch.tensor(test_md).long()
        test_set['mm_mdG']=self.GIP_sim(test_set['md']).float()
        test_set['dd_dmG']=self.GIP_sim(test_set['md'].T).float()
        test_set['mm_mdF']=self.Functional_sim(test_set['md'],common_set['dd_sem'],self.params.device,64).float()
        test_set['dd_dmF']=self.Functional_sim(test_set['md'].T,common_set['mm_seq'],self.params.device,64).float()

        # torch.save(test_set,'./mdd/test_set.pkl')
        print('test_set saved')

        for k in range(self.params.kfolds):
            xy_train,xy_valid=train_xy[train_idx[k],:],train_xy[valid_idx[k],:]
            label_train,label_valid=train_label[train_idx[k]],train_label[valid_idx[k]]
            train_md=np.zeros(self.md.shape)
            train_md[xy_train[:,0],xy_train[:,1]]= label_train
            train_set['edge_train_%d'%k]=torch.tensor(xy_train).long()
            train_set['label_train_%d'%k]=torch.tensor(label_train).long()
            train_set['edge_valid_%d'%k]=torch.tensor(xy_valid).long()
            train_set['label_valid_%d'%k]=torch.tensor(label_valid).long()
            train_set['md_%d'%k]=torch.tensor(train_md).long()

            train_set['mm_mdG_%d'%k]=self.GIP_sim(train_set['md_%d'%k]).float()
            train_set['dd_dmG_%d'%k]=self.GIP_sim(train_set['md_%d'%k].T).float()
            train_set['mm_mdF_%d'%k]=self.Functional_sim(train_set['md_%d'%k],common_set['dd_sem'],self.params.device,64).float()
            train_set['dd_dmF_%d'%k]=self.Functional_sim(train_set['md_%d'%k].T,common_set['mm_seq'],self.params.device,64).float()

        # torch.save(train_set,'./mdd/train_set.pkl')
        print('train_set saved')

dl = dataloader(args) 