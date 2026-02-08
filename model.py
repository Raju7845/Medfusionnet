import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel

class CMAFModule(nn.Module):
    """Cross-Modal Attention Fusion (CMAF) - Section 3.2.3"""
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )

    def forward(self, img_feat, text_feat):
        # Cross-modal Attention: Image queries Text context
        attn_out, _ = self.mha(img_feat, text_feat, text_feat)
        x = self.ln1(img_feat + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x

class MedFusionNet(nn.Module):
    def __init__(self, num_classes_list=[2, 3, 2]):
        super().__init__()
        # Visual Stream (ViT) and Text Stream (DistilBERT)
        self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        
        self.fusion = CMAFModule(dim=768)
        
        # Multi-task Classification Heads
        self.bc_head = nn.Linear(768, num_classes_list[0]) # Breast Cancer
        self.cc_head = nn.Linear(768, num_classes_list[1]) # Cervical Cancer
        self.pcos_head = nn.Linear(768, num_classes_list[2]) # PCOS

    def forward(self, images, input_ids, attention_mask):
        img_feats = self.img_encoder(images).last_hidden_state
        text_feats = self.text_encoder(input_ids, attention_mask).last_hidden_state
        
        fused = self.fusion(img_feats, text_feats)
        cls_token = fused[:, 0, :] # Extracting fused CLS representation
        
        return {
            'bc': self.bc_head(cls_token),
            'cc': self.cc_head(cls_token),
            'pcos': self.pcos_head(cls_token)
        }