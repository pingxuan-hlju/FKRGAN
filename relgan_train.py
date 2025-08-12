import os
import torch
import torch.nn as nn
import numpy as np
import model0527
from parameters import args
from sklearn.model_selection import KFold
import random
from early_stopping import EarlyStopping
from torch.utils.data import dataset,dataloader

# 随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)    
torch.backends.cudnn.deterministic = True


#### 生成数据
class load_data(nn.Module):
    def __init__(self, args):
        super(load_data, self).__init__()
        self.args = args
        self.md = np.load(args.miRNA_disease_adj_path)
        self.common_set=torch.load('./mdd/common_set.pkl')
        self.train_set=torch.load('./mdd/train_set.pkl')
        self.triples4lable = self.data_set()
    # 阈值处理
    def threshold_todo_idex(self):
        mirna_sim_mat = self.common_set['mm_seq'] >= self.args.threshold
        dise_sim_mat = self.common_set['dd_sem'] >= self.args.threshold
        asso_mat = torch.tensor(self.md).float()

        # 获取索引值
        mirna_inter_adj = mirna_sim_mat.nonzero()
        dise_inter_adj = dise_sim_mat.nonzero()
        mirna_dise_adj = asso_mat.nonzero()
        # 添加offset
        miRNA_offset, disease_offset = 0, 1245
        mirna_inter_adj = mirna_inter_adj + torch.tensor([miRNA_offset,miRNA_offset])
        dise_inter_adj = dise_inter_adj + torch.tensor([disease_offset,disease_offset])
        mirna_dise_adj = mirna_dise_adj + torch.tensor([miRNA_offset,disease_offset])
        
        return mirna_inter_adj,dise_inter_adj,mirna_dise_adj

    # 按照train_ratio, valid_ratio, test_ratio 划分索引，返回集合
    # mirna_dise_rel, 0;  mirna_inter_rel, 1; dise_inter_rel, 2; dise_mirna_rel 3
    def split_dataset(self,data, trt, vat, tet, rel= 0):
        rand_num= np.random.rand(len(data))
        train, valid, test= data[rand_num<= trt], data[(rand_num> trt)* (rand_num<= (trt+ vat))], data[rand_num> (trt+ vat)]
        train_rels, valid_rels, test_rels= np.ones(len(train))* rel, np.ones(len(valid))* rel, np.ones(len(test))* rel
        return np.insert(train, 2, train_rels, axis= 1), np.insert(valid, 2, valid_rels, axis= 1), np.insert(test, 2, test_rels, axis= 1)       

    # 划分五折数据
    def data_set(self):
        mirna_inter_adj,dise_inter_adj,mirna_dise_adj = self.threshold_todo_idex()
        mirna_disease_train, _, _= self.split_dataset(mirna_dise_adj, 0.8, 0.1, 0.1, 0)
        mirna_inter_train,_,_= self.split_dataset(mirna_inter_adj, 1, 0, 0, 1)
        dise_inter_train,_,_= self.split_dataset(dise_inter_adj, 1, 0, 0, 2)
        train_xy_inter = torch.cat([mirna_disease_train,mirna_inter_train,dise_inter_train],dim=0)
        train_label = torch.ones(train_xy_inter.shape[0])
        train_rd=np.random.permutation(train_xy_inter.shape[0])
        train_xy_inter,train_label=train_xy_inter[train_rd,:],train_label[train_rd]
        gan_rel_set= {}
        kf = KFold(n_splits=self.args.kfolds, shuffle=True, random_state=self.args.seed)
        k = 0
        for train_index, _ in kf.split(train_xy_inter):
            # print(f'第{k + 1}折')
            train_xy = train_xy_inter[train_index]
            train_label = torch.ones(train_xy.shape[0])
            train_rd=np.random.permutation(train_xy.shape[0])
            train_xy,train_label=train_xy[train_rd,:],train_label[train_rd]        
            gan_rel_set['triples%d'%k] = train_xy
            gan_rel_set['label%d'%k] = train_label
            # print(train_xy.shape)
            k += 1
        torch.save(gan_rel_set, './savedata/gan_rel_set.pth')     
        print('gan_rel_set saved')
        return gan_rel_set

#### run
def train_data(args,generator_net,discriminator_net,trainLoader,k):
    # Optimizers
    optimizer_G = torch.optim.Adam(generator_net.parameters(), lr=args.lr, weight_decay=args.weight)
    optimizer_D = torch.optim.Adam(discriminator_net.parameters(), lr=args.lr, weight_decay=args.weight)
    # 早停
    ES=EarlyStopping("./models/GAN_fold_%d.pkl"%k)
    loss_fun = nn.BCELoss()
    
    for ep in range(args.epoch_relgan):
        d_epoch_loss = 0
        g_epoch_loss = 0
        generator_net.train(),discriminator_net.train()
        for data,label in trainLoader:
            left,right,rel_idx = data[:,0].to(args.device),data[:,1].to(args.device),data[:,2].to(args.device)
            rela_label = label.to(args.device)
            fake_label = torch.zeros_like(label).to(args.device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            org_embeds,gan_emb = generator_net(left,right,rel_idx)
            fake_output = discriminator_net(org_embeds[left],gan_emb,rel_idx)
            g_fake_loss = loss_fun(fake_output,rela_label) 
            # 正则化项
            g_reg_loss = sum((param**2).sum() for param in generator_net.parameters())
            g_loss = g_fake_loss + args.lambda_reg *g_reg_loss
            g_epoch_loss += g_loss.item()
            # 反向传播
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # 真正节点对的loss
            real_output = discriminator_net(org_embeds[left].detach(),org_embeds[right].detach(),rel_idx)
            d_real_loss = loss_fun(real_output,rela_label)
            # 生成器生成的节点的loss
            fake_output = discriminator_net(org_embeds[left].detach(),gan_emb.detach(),rel_idx)
            d_fake_loss = loss_fun(fake_output,fake_label)
            # 正则化项
            d_reg_loss = sum((param**2).sum() for param in discriminator_net.parameters())
            d_loss = (d_real_loss + d_fake_loss) / 2 + args.lambda_reg *d_reg_loss
            d_epoch_loss += d_loss.item()
            d_loss.backward()
            optimizer_D.step()
        print(f'Cross: {k+1}, epoch: {ep+1}, D loss: {d_epoch_loss:.5f}, G loss: {g_epoch_loss:.5f}')
        # print("Real output avg:", real_output.mean().item())
        # print("Fake output avg:", fake_output.mean().item())    
        if ES(g_loss.item(), generator_net):
                print("Early stopping at epoch", ep)
                break


def generate_gan_features(FBtrain_set,FBtest_set,args):
    GAN_embeds_train, GAN_embeds_test= {},{}
    for k in range(args.kfolds):
        HeG_emb_train = FBtrain_set['HeG_emb_train_%d'%k].to(args.device)
        HoG4rfam_emb_train = FBtrain_set['HoG4rfam_emb_train_%d'%k].to(args.device)
        HoG4cluster_emb_train = FBtrain_set['HoG4cluster_emb_train_%d'%k].to(args.device)
        HoG4targets_emb_train = FBtrain_set['HoG4targets_emb_train_%d'%k].to(args.device)
        # 加载GAN生成器模型参数
        generator_net = model0527.Generator(HeG_emb_train,HoG4rfam_emb_train,HoG4cluster_emb_train,HoG4targets_emb_train,args.in_dim, args.hidden_dim).to(args.device)
        generator_net.load_state_dict(torch.load('./models/GAN_fold_%d.pkl'%k))
        for param in generator_net.parameters():
            param.requires_grad = False
        # 所有节点的的索引值
        left, right =  torch.arange(0, 1245).long().to(args.device), torch.arange(0, 2077).long().to(args.device)
        # r0: MD;  r1:DM
        r0_idx = torch.zeros_like(left)
        r1_idx = torch.zeros_like(right)+3
        with torch.no_grad():
            _,gen_feature_miran = generator_net(left,right+1245,r0_idx)
            _,gen_feature_dise = generator_net(right+1245,left,r1_idx)
            GAN_embeds_train['gan_Mfeat_%d'%k]= gen_feature_miran.cpu()
            GAN_embeds_train['gan_Dfeat_%d'%k]= gen_feature_dise.cpu()
    
    torch.save(GAN_embeds_train,'./mdd/GAN_train_set.pkl')
    print('GAN_embeds_train_set saved')

    # 保存训练集所生成的特征
    test_Mfeat_ls,test_Dfeat_ls = [],[]
    HoG4rfam_emb = FBtest_set['HoG4rfam_emb_test'].to(args.device)
    HoG4cluster_emb = FBtest_set['HoG4cluster_emb_test'].to(args.device)    
    HoG4targets_emb = FBtest_set['HoG4targets_emb_test'].to(args.device)
    HeG_emb = FBtest_set['HeG_emb_test'].to(args.device)
    net = model0527.Generator(HeG_emb,HoG4rfam_emb,HoG4cluster_emb,HoG4targets_emb,args.in_dim, args.hidden_dim).to(args.device)
    for i in range(args.kfolds):
        net.load_state_dict(torch.load('./models/GAN_fold_%d.pkl'%i))
        for param in net.parameters():
            param.requires_grad = False
        # 所有节点的的索引值
        left, right =  torch.arange(0, 1245).long().to(args.device), torch.arange(0, 2077).long().to(args.device)
        # r0: MD;  r1:DM
        r0_idx = torch.zeros_like(left)
        r1_idx = torch.zeros_like(right)+3
        net.eval()
        with torch.no_grad():
            _,gen_feature_miran = net(left,right+1245,r0_idx)
            _,gen_feature_dise = net(right+1245,left,r1_idx)
            test_Mfeat_ls.append(gen_feature_miran)
            test_Dfeat_ls.append(gen_feature_dise)

    avg_Mfeat = torch.stack(test_Mfeat_ls).mean(0)
    avg_Dfeat = torch.stack(test_Dfeat_ls).mean(0)

    GAN_embeds_test['GAN_Mfeat_test'] = avg_Mfeat.cpu()
    GAN_embeds_test['GAN_Dfeat_test'] = avg_Dfeat.cpu()

    torch.save(GAN_embeds_test,'./mdd/GAN_test_set.pkl')
    print('GAN_embeds_test_set')


if __name__ == "__main__":
    # dl = load_data(args)
    FBtrain_set = torch.load('./mdd/FBtrain_set.pkl')
    FBtest_set = torch.load('./mdd/FBtest_set.pkl')
    gan_rel_set = torch.load('./savedata/gan_rel_set.pth')
    for k in range(args.kfolds): 
        HeG_emb_train = FBtrain_set['HeG_emb_train_%d'%k].to(args.device)
        HoG4rfam_emb_train = FBtrain_set['HoG4rfam_emb_train_%d'%k].to(args.device)
        HoG4cluster_emb_train = FBtrain_set['HoG4cluster_emb_train_%d'%k].to(args.device)
        # print(HeG_emb_train.shape,HoG4rfam_emb_train.shape,HoG4cluster_emb_train.shape)
        HoG4targets_emb_train = FBtrain_set['HoG4targets_emb_train_%d'%k].to(args.device)
        # 划分数据 
        trainLoader = dataloader.DataLoader(model0527.ALDataset(gan_rel_set['triples%d'%k], gan_rel_set['label%d'%k]), batch_size=8192, shuffle=True, num_workers=0)
        # 实例化模型
        generator_net = model0527.Generator(HeG_emb_train,HoG4rfam_emb_train,HoG4cluster_emb_train,HoG4targets_emb_train,args.in_dim, args.hidden_dim).to(args.device)
        discriminator_net = model0527.Discriminator(args.in_dim, args.hidden_dim).to(args.device)
        train_data(args,generator_net,discriminator_net,trainLoader,k)
    generate_gan_features(FBtrain_set,FBtest_set,args)    

