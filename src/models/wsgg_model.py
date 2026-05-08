import torch

class WSGGModel(torch.nn.Module):
    def __init__(self, Ng=5, Exp=4, Pt=1.0, X=0.2, T_ref=1400):
        """
        WSGG模型初始化
        
        参数:
        Ng (int): 灰气体总数
        Exp (int): 多项式阶数
        Pt (float): 总压 (bar)
        X (float): 参与性气体摩尔分数
        T_ref (float): 参考温度 (K)
        """
        super().__init__()
        # Ng现在包括所有气体：第一个是透明气体，其余是灰气体
        assert Ng >= 2, "Ng必须至少为2（一个透明气体+两个灰气体）"
        self.Ng = Ng  # 灰气体数
        # self.gray_count = Ng - 1  # 灰气体数
        self.Exp = Exp
        self.Pt = Pt
        self.X = X
        self.T_ref = T_ref
        
        # 初始化所有Ng种气体的参数
        # self.beta = torch.nn.Parameter(torch.randn(Ng+1, 15) * 0.1)  # 权重参数
        # self.gamma = torch.nn.Parameter(torch.randn(Ng+1, Exp+1) * 0.1)  # 吸收系数参数
        self.beta = torch.nn.Parameter(torch.randn(Ng+1, 15))  # 权重参数
        self.gamma = torch.nn.Parameter(torch.randn(Ng+1, Exp+1))  # 吸收系数参数

    def compute_a_coefficients(self, Tr, Mr):
        # 生成多项式特征矩阵
        features = torch.stack([
            Tr**p * Mr**q 
            for p in range(5) 
            for q in range(5) 
            if p + q <= 4
        ], dim=-1)
        
        # 计算所有气体的权重
        logits = torch.einsum('nf,...f->n...', self.beta, features)
        
        # 使用softmax确保所有气体权重和为1
        return torch.nn.functional.softmax(logits, dim=0)

    def compute_K_coefficients(self, Mr):
        # 计算所有气体的原始吸收系数
        powers = torch.arange(self.Exp, -1, -1, device=Mr.device, dtype=torch.float32)
        mr_powers = Mr.unsqueeze(-1) ** powers
        poly_val = torch.einsum('ne,be->nb', self.gamma, mr_powers)
        
        # 使用softplus确保吸收系数非负
        K = torch.nn.functional.softplus(poly_val)
        
        # 固定第一个气体（透明气体）的吸收系数为0
        # 使用mask将第一个气体的吸收系数设为零
        mask = torch.ones(self.Ng+1, device=K.device, dtype=torch.bool)
        mask[0] = False  # 第一个气体（透明气体）不参与吸收
        
        # 应用mask: 只有灰气体保留计算得到的吸收系数
        return K * mask.unsqueeze(-1).float()

    def forward(self, Mr, Tr, L):
        # 获取所有气体的权重（包括透明气体）
        a = self.compute_a_coefficients(Tr, Mr)
        
        # 获取所有气体的吸收系数（透明气体的吸收系数为0）
        K = self.compute_K_coefficients(Mr)
        
        # 计算光学厚度
        optical_thickness = K * self.Pt * self.X * L
        
        # 计算发射率（透明气体部分自动贡献为0）
        epsilon = torch.sum(a * (1 - torch.exp(-optical_thickness)), dim=0)
        
        return epsilon