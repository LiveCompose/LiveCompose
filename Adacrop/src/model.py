import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = resnet # 以resnet50作为backbone
        self.backbone.fc = nn.Identity() # 去掉全连接层
        feat_dim = 2048  # ResNet50特征维度

        # Actor分支
        # 输入：图像特征 + 状态（归一化后的 (cx, cy, w, h)）
        # 输出：动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(feat_dim + 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_actions)
        )
        
        # Critic分支
        # 输入：图像特征 + 状态（归一化后的 (cx, cy, w, h)）
        # 输出：单一价值
        self.critic = nn.Sequential(
            nn.Linear(feat_dim + 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1) # 输出单一价值
        )
        
        # 预训练用的bbox回归头
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

        if isinstance(self.actor[-1], nn.Linear):
            nn.init.zeros_(self.actor[-1].bias)

    def extract_feats(self, img_tensor):
        """
        img_tensor: [B,3,H,W]
        returns: [B, feat_dim]
        """
        if self.training:
            # 训练模式：保留梯度，但冻结BN层统计
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()  # BN层使用预训练统计
            features = self.backbone(img_tensor)
        else:
            # 推理模式：禁用梯度计算
            with torch.no_grad():
                self.backbone.eval()
                features = self.backbone(img_tensor)
    
        return features
    
    def backbone_forward(self, x):
        feats = self.extract_feats(x)  # [B, feat_dim]
        return self.bbox_head(feats)   # [B, 4]
    
    def forward(self, img_tensor, state):
        """
        img_tensor: [B,3,H,W]
        state:      [B,4]  归一化后的 (cx, cy, w, h)
        """
        img_feats = self.extract_feats(img_tensor) # 将图像的高维像素数据转换成一个低维的特征向量
        # 将图像特征 img_feats 和状态信息 state 在维度1上进行拼接
        x = torch.cat([img_feats, state], dim=1)

        action_logits = self.actor(x) #导入Actor网络
        action_logits = action_logits - action_logits.max(dim=1, keepdim=True)[0]               # [B, n_actions]
        action_probs = F.softmax(action_logits, dim=1)  
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)

        epsilon = 1e-8
        action_probs = action_probs + epsilon
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        
        value = self.critic(x)                       # [B, 1]
        
        return action_probs, value
