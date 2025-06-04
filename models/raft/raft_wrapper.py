import torch
import numpy as np
import cv2
import sys
import os
from types import SimpleNamespace

# RAFT kodları ve model dosyası aynı klasörde olmalı
sys.path.append(os.path.dirname(__file__))
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

class RAFTOpticalFlow:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        args = SimpleNamespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0
        )
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.module
        self.model.to(device)
        self.model.eval()

    def calc_flow(self, img1, img2):
        # img1, img2: np.array, RGB, [H,W,3], 0-255
        img1 = torch.from_numpy(img1).permute(2,0,1).float().unsqueeze(0) / 255.0
        img2 = torch.from_numpy(img2).permute(2,0,1).float().unsqueeze(0) / 255.0
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        with torch.no_grad():
            flow_low, flow_up = self.model(img1.to(self.device), img2.to(self.device), iters=20, test_mode=True)
        flow = padder.unpad(flow_up[0]).cpu().numpy().transpose(1,2,0)
        return flow 