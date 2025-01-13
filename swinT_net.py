import torch.nn as nn
import timm
import torch

# class CustomSwinTransformer(nn.Module):
#     def __init__(self, num_classes, pretrained_path='model\swin_tiny_patch4_window7_224.pth'):
#         super(CustomSwinTransformer, self).__init__()
#         self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
        
#         if pretrained_path:
#             state_dict = torch.load(pretrained_path, map_location=torch.device('cpu')) 
#             self.model.load_state_dict(state_dict, strict=False) 
#     def forward(self, x):
#         return self.model(x)

class CustomSwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super(CustomSwinTransformer, self).__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)