import torch
from torch import nn
from parameters import args
import torch.nn.functional as F
from torch.utils.data import dataset,dataloader
import math

# 划分数据集合
class ALDataset(dataset.Dataset):
    def __init__(self, edges, labels):
        self.Data = edges
        self.Label = labels
    def __len__(self):
        return len(self.Label)
    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label

#  @ gcn 层
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        """
        GCN Layer: H = ReLU(A_hat * X * W)
        """
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # W 矩阵
    def forward(self,X,A_hat):
        emb = self.linear(torch.matmul(A_hat, X))
        return emb  # H = ReLU(A_hat * X * W)

# @ 定义一维离散傅里叶变换编码层
class EDFConv(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(EDFConv, self).__init__()
        self.DF_gcn = GCNLayer(hid_dim*2,2*out_dim)
        self.act1 = nn.LeakyReLU()
        self.n1=nn.LayerNorm(2*out_dim)
    def forward(self, x,ADJ):
        x = x.unsqueeze(1)
        batch, c, w = x.size()
        # 一维实数傅里叶变换
        ffted = torch.fft.rfft(x, dim=-1, norm='ortho')  # [batch, c, freq]
        # 拆分实部和虚部并合并
        real = torch.real(ffted).unsqueeze(-1)  # [batch, c, freq, 1]
        imag = torch.imag(ffted).unsqueeze(-1)  # [batch, c, freq, 1]
        ffted = torch.cat((real, imag), dim=-1)  # [batch, c, freq, 2]
        # 调整为 Conv2d 输入格式
        ffted = ffted.permute(0, 1, 3, 2).contiguous()  # [batch, c, 2, freq]
        ffted = ffted.view(batch, -1, 1, ffted.size(-1))  # [batch, c*2, 1, freq]
        # GCN、BN、ReLU
        ffted = self.n1(self.act1(self.DF_gcn(ffted.view(batch,-1),ADJ)))
        # 一维逆傅里叶变换
        ffted = ffted.view(batch, -1, 2, int(ffted.size(-1)/2))  # [batch, out_c, 2, freq]
        ffted = ffted.permute(0, 1, 3, 2).contiguous()    # [batch, out_c, freq, 2]
        ffted = torch.view_as_complex(ffted)              # [batch, out_c, freq]
        output = torch.fft.irfft(ffted, n=w, dim=-1, norm='ortho')  # [batch, out_c, w]
        return output  

# @ 一维傅里叶变换解码器
class DDFConv(nn.Module):
    def __init__(self):
        super(DDFConv, self).__init__()
        self.bn = nn.BatchNorm2d(1 * 2)
        self.DF_gcn1 = GCNLayer(130,1024)
        self.DF_gcn2 = GCNLayer(1024,3322)
        self.n1=nn.LayerNorm(1024)
        self.n2=nn.LayerNorm(3322)
        self.act1=nn.LeakyReLU()
    def forward(self, x, ADJ):
        x = x.unsqueeze(1)
        batch, c, w = x.size()
        # 傅里叶变换
        ffted = torch.fft.rfft(x, dim=-1, norm='ortho')
        real = torch.real(ffted).unsqueeze(-1)
        imag = torch.imag(ffted).unsqueeze(-1)
        ffted = torch.cat((real, imag), dim=-1)
        ffted = ffted.permute(0, 1, 3, 2).contiguous()
        ffted = ffted.view(batch, -1, 1, ffted.size(-1))
        # 频域空间的解码器
        ffted = self.n1(self.act1(self.DF_gcn1(ffted.view(batch,-1),ADJ)))
        ffted = self.n2(self.act1(self.DF_gcn2(ffted,ADJ)))
        # 傅里叶逆变换
        ffted = ffted.view(batch, -1, 2, int(ffted.size(-1)/2))
        ffted = ffted.permute(0, 1, 3, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft(ffted, n=w, dim=-1, norm='ortho')
        return output          

#   @ 图卷积自编码器
class feature_Block(nn.Module):
    def __init__(self, in_dim, hid_dim,out_dim):
        super(feature_Block, self).__init__()
        #  编码层 
        self.encoder_gcn1 = GCNLayer(in_dim, hid_dim)
        self.encoder_gcn2 = GCNLayer(hid_dim,out_dim)
        # self.encoder_dft1 = EDFConv(4,1024)
        self.encoder_dft1 = EDFConv(in_dim//2 + 1,1024)
        self.encoder_dft2 = EDFConv(513,128)
        # 解码层
        self.decoder_gcn1 = GCNLayer(out_dim, hid_dim)
        self.decoder_gcn2 = GCNLayer(hid_dim,in_dim)       
        self.decoder_dfts = DDFConv()
        # 对齐
        self.par = nn.Parameter(torch.randn(2))
        self.alpha = nn.Parameter(torch.randn(1))
        self.fc1 = nn.Linear(in_dim,1024,bias=False)
        self.fc2 = nn.Linear(1024,128,bias=False)
        self.fc3 = nn.Linear(128,in_dim,bias=False)
        # 激活函数
        self.act1=nn.LeakyReLU()
    def forward(self,feats,adj,left,right,method):
        a,b = F.softmax(self.par,dim=0)
        #  编码层 
        emb1_ES = self.act1(self.encoder_gcn1(feats, adj))
        emb1_DF = self.fc1(self.encoder_dft1(feats,adj).squeeze())
        emb_encodeer1 = (1-a) * emb1_ES + a * emb1_DF

        emb2_ES = self.act1(self.encoder_gcn2(emb_encodeer1,adj))
        emb2_DF = self.fc2(self.encoder_dft2(emb1_DF,adj).squeeze())
        emb_encodeer2 = (1-b) * emb2_ES + b * emb2_DF
        
        # 解码层
        # 欧式空间
        emb1_ES_dec = self.act1(self.decoder_gcn1(emb_encodeer2,adj))
        emb2_ES_dec = self.act1(self.decoder_gcn2(emb1_ES_dec,adj))
        # 频域空间
        emb_DF_dec = self.fc3(self.decoder_dfts(emb2_DF,adj).squeeze())
        embed = (1-self.alpha) * emb2_ES_dec + self.alpha * emb_DF_dec
        emb_decoder2 = F.sigmoid(embed)
        # 重构损失函数
        if method == "HeG4embed":   
            X_batch = torch.cat([feats[left],feats[right]],dim=0)
            emb_batch = torch.cat([emb_decoder2[left],emb_decoder2[right]],dim=0)
        
        elif method == "miRNA_rfam4embed" or method == "miRNA_cluster4embed" or method == "miRNA_targets4embed":
            X_batch = feats[left]
            emb_batch = emb_decoder2[left]
        loss = F.mse_loss(X_batch, emb_batch, reduction='mean')       
        return emb_encodeer2, loss
    
# @ 特征级注意力机制
class fusion_Block(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_views=4):
        super(fusion_Block,self).__init__()
        # 特征级注意力机制
        self.num_views = num_views
        self.W_att_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(self.num_views)])
        self.h_att_list = nn.ParameterList([nn.Parameter(torch.Tensor(hidden_dim)) for _ in range(self.num_views)])


        # 初始化
        for h in self.h_att_list:
            nn.init.xavier_uniform_(h.unsqueeze(0))

    # def forward(self,X_topo, X_family, X_cluster,X_targets):
    def forward(self,X_topo,X_family,X_cluster,X_targets):
        # [N_miRNA + N_disease, layer_dim] 
        dise_topo = X_topo[1245:,:]
        # [N_miRNA, layer_dim] 
        miRNA_topo = X_topo[:1245,:]
        # 同质图 Homogeneous graph
        miRNA_targets = X_targets
        miRNA_family, miRNA_cluster = X_family, X_cluster
        # [N_miRNA, 2, layer_dim]
        miRNA_stack = torch.stack([miRNA_topo,miRNA_family, miRNA_cluster,miRNA_targets], dim=1)
        # [N_miRNA, 2, hidden_dim]：存每个视角的变换结果
        projections = []
        for v in range(self.num_views):
            #F_stack[:, v]* W_att_list[v] = [N_miRNA, layer_dim] * [layer_dim, hidden_dim]  ->   [N_miRNA, hidden_dim]
            proj = torch.tanh(self.W_att_list[v](miRNA_stack[:, v]))  
            projections.append(proj)

        # [N_miRNA, 2, hidden_dim]
        proj_all = torch.stack(projections, dim=1)

        # [2, hidden_dim] -> unsqueeze(0) -> [1, 2, hidden_dim]
        h_stack = torch.stack([h for h in self.h_att_list], dim=0).unsqueeze(0)

        # 点积：[N_miRNA, 2, hidden_dim] · [1, 2, hidden_dim] -> [N_miRNA, 2]
        scores = torch.sum(proj_all * h_stack, dim=-1)  # [N_miRNA, 2]
        alpha = F.softmax(scores, dim=-1)               # [N_miRNA, 2]

        # 注意力加权求和：[N_miRNA, 2, 1] * [N_miRNA, 2, layer_dim] -> [N_miRNA, 2, layer_dim]
        miRNA_fused = torch.sum(alpha.unsqueeze(-1) * miRNA_stack, dim=1)  # [N_miRNA, layer_dim]

        return torch.cat([miRNA_fused,dise_topo], dim=0)  # [N_miRNA + N_disease, layer_dim] 


# @ KANLayer
class KANLayer(nn.Module):
    def __init__(
            self,in_dim, out_dim, grid_size=5,spline_order=3, grid_range=[-1, 1], 
            base_activation=torch.nn.SiLU, scale_noise=0.1,scale_base=1.0,
            scale_spline=1.0,enable_standalone_scale_spline=True,grid_eps=0.02
            ):
        super(KANLayer,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0] 
            )
            .expand(in_dim, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_dim, in_dim, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.scale_noise = scale_noise 
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_dim, self.out_dim)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    
    def b_splines(self, x):
        #定义 (in_features, grid_size + 2 * spline_order + 1)
        grid: torch.Tensor = (self.grid) 
        x = x.unsqueeze(-1)
        # 计算 0 阶 B-样条基函数值
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # 递归计算 k-1 阶B-样条基函数值
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()
    def curve2coeff(self, x, y):
        # 计算 B-样条基函数
        # (in_features, batch_size, grid_size + spline_order)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        # 使用最小二乘法求解线性方程组
        # (in_features, grid_size + spline_order, out_features)
        solution = torch.linalg.lstsq(A, B).solution
        # 调整结果的维度顺序
        # (out_features, in_features, grid_size + spline_order)
        result = solution.permute(2,0,1)
        return result.contiguous()  
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0)
    def forward(self, x):
        # 计算基础线性层的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # 计算分段多项式线性层的输出
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_dim, -1),)
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        batch = x.size(0)
        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

class KAN(nn.Module):
    def __init__(
        self, layers_hidden,grid_size=5,spline_order=3,grid_range=[-1, 1],
        scale_noise=0.1,scale_base=1.0,scale_spline=1.0,grid_eps=0.02,
        base_activation=torch.nn.SiLU):
        super(KAN,self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLayer(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
    def forward(self, x: torch.Tensor, update_grid=False):    
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x
            
  
#  关系敏感对抗生成网络
# @ 生成器
class Generator(nn.Module):
    def __init__(self,X_topo,X_family,X_cluster,X_targets,in_dim, hd_dim):
    # def __init__(self,X_topo,X_targets,in_dim, hd_dim):
        super(Generator, self).__init__()
        self.X_topo,self.X_targets = X_topo,X_targets
        self.X_family,self.X_cluster = X_family,X_cluster
        self.fusion_Block = fusion_Block(args.in_dim, args.hidden_dim, 4)
        # mirna_dise_rel, 0;  mirna_inter_rel, 1; dise_inter_rel, 2; dise_mirna_rel 3
        A_r = [torch.randn(in_dim, in_dim) for _ in range(4)]
        self.A_r_tensor = nn.Parameter(torch.stack(A_r))  # [4, in_dim, in_dim]
        self.fc_block = KAN([in_dim,hd_dim,in_dim])


    def forward(self,left,right,r_idx):
        # 融合三个特征矩阵为一个特征矩阵
        fusion_embed = self.fusion_Block(self.X_topo,self.X_family,self.X_cluster,self.X_targets)
        # fusion_embed = self.X_topo
        # 根据r_idx找到对应的关系矩阵
        A_r_batch = self.A_r_tensor[r_idx]
        # 该关系下的真实头节点的特征
        emb_u = fusion_embed[left]
        # 计算均值
        mean = torch.bmm(emb_u.unsqueeze(1), A_r_batch).squeeze(1)  # [batch_size, in_dim]
        # 计算协方差矩阵
        cov = torch.eye(mean.shape[1], device=mean.device)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        # 获得高斯噪声矩阵
        emb = dist.rsample()
        # print(emb_u.shape,emb.shape)
        # Gen_emb = emb_u + emb
        Gen_emb = emb_u * emb
        # Gen_emb = emb_u + emb.cuda()      # (batch, feat_dim)
        e_v_fake = self.fc_block(Gen_emb)     # (batch, feat_dim)
        return fusion_embed,e_v_fake

# @ 判别器
class Discriminator(nn.Module):
    def __init__(self,in_dim,hd_dim):
        super(Discriminator,self).__init__()
        # mirna_dise_rel, 0;  mirna_inter_rel, 1; dise_inter_rel, 2; dise_mirna_rel 3
        A_r = [torch.randn(in_dim, in_dim) for _ in range(4)]
        self.A_r_tensor = nn.Parameter(torch.stack(A_r))  # [4, in_dim, in_dim]
        self.fc_block = KAN([in_dim,hd_dim,1])
        
    def forward(self,emb1,emb2, r_idx):
            A_r_batch = self.A_r_tensor[r_idx]
            # Hadamard product
            h = emb1 * emb2  # (batch, in_dim)
            # 关系矩阵作用
            h = torch.bmm(h.unsqueeze(1), A_r_batch).squeeze(1)  # [batch_size, in_dim]
            # 两层前馈神经网络
            out = self.fc_block(h)  # (batch, 1)
            score = torch.sigmoid(out).squeeze()  # (batch,) if binary classification
            return score
    
# @ CNN 
class CNN_Model(nn.Module):
    def __init__(self,ks):
        super(CNN_Model, self).__init__()
        self.mfea=None
        self.dfea=None
        self.c1=nn.Conv1d(2,8,kernel_size=ks,stride=1,padding=0)
        self.s1=nn.MaxPool1d(kernel_size=ks)
        self.c2=nn.Conv1d(8,16,kernel_size=ks,stride=1,padding=0)
        self.l1=nn.Linear(16*69,64)
        self.l2=nn.Linear(64,2)
        self.act1=nn.LeakyReLU()
        self.cost=nn.CrossEntropyLoss()
    def build_fea(self,mm,md,dd,embedM,embedD):
        self.mfea=torch.cat([mm,md,embedM],dim=1)
        self.dfea=torch.cat([md.t(),dd,embedD],dim=1)
    def forward(self,x,y):
        hs=torch.stack([self.mfea[x[:,0]],self.dfea[x[:,1]]],dim=1)
        hs=self.s1(self.act1(self.c1(hs)))
        hs=self.s1(self.act1(self.c2(hs)))
        # a = hs.reshape(hs.shape[0],-1)
        # print(a.shape)
        hs=self.act1(self.l1(hs.reshape(hs.shape[0],-1)))
        hs=self.l2(hs)
        return hs,self.cost(hs,y)