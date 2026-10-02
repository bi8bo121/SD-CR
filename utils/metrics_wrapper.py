import numpy as np
import cv2
from utils.metrics_tool import Evaluator 

class FusionMetricsTracker:
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()

    def reset(self):
        self.metrics_sum = {
            'SD': 0.0,  
            'EN': 0.0, 
            'AG': 0.0,  
            'SF': 0.0  
        }
        self.count = 0

    def _tensor_to_gray_numpy(self, tensor_img):

        imgs_np = tensor_img.permute(0, 2, 3, 1).cpu().detach().numpy() 
        
        gray_imgs = []
        for img in imgs_np:
         
            img = np.clip(img * 255.0, 0, 255).astype(np.float32) 

            if img.ndim == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim == 3 and img.shape[2] == 1:
                gray = img[:, :, 0]
            else:
                gray = img
            
            gray_imgs.append(gray)
            
        return gray_imgs

    def update(self, fused_tensor):
       
        f_list = self._tensor_to_gray_numpy(fused_tensor)
        
        batch_size = len(f_list)
        
        for i in range(batch_size):
            F = f_list[i]
            
            try:
                self.metrics_sum['SD'] += Evaluator.SD(F)
                self.metrics_sum['EN'] += Evaluator.EN(F)
                self.metrics_sum['AG'] += Evaluator.AG(F)
                self.metrics_sum['SF'] += Evaluator.SF(F)
            except Exception as e:
                print(f"Metric calculation error: {e}")
                
        self.count += batch_size

    def compute(self):
        if self.count == 0:
            return {k: 0.0 for k in self.metrics_sum}
        return {k: v / self.count for k, v in self.metrics_sum.items()}