import cv2
import torch
from hub_models import midas, midas_transform
import torch.nn.functional as F
def get_depth_heatmap(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = midas_transform(rgb).to('cuda')
    if tensor.ndim == 5:
        tensor = tensor.squeeze(0)
    with torch.no_grad():
        pred = midas(tensor)
    pred = F.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth = pred.cpu().numpy()
    depth_norm = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype('uint8')
    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    return heatmap
