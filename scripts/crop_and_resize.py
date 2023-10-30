import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--src_root", type=str, required=True)
parser.add_argument("--dst_root", type=str, required=True)
parser.add_argument("--desired_size", type=int, default=512)
parser.add_argument("--num_data", type=int, default=200)

args = parser.parse_args()

os.makedirs(args.dst_root, exist_ok=True)
fnames = sorted(os.listdir(args.src_root))
idx = np.linspace(0, len(fnames)-1, args.num_data).astype(np.int32)
fnames = [f for i, f in enumerate(fnames) if i in idx]

for f in tqdm(fnames):
    img = cv2.imread(os.path.join(args.src_root, f))
    H, W = img.shape[:2]
    L = min(H, W)

    if H != W:
        Hc, Wc = (H-L)//2, (W-L)//2
        img = img[Hc:Hc+L, Wc:Wc+L]
    
    if L > args.desired_size:
        img = cv2.resize(img, dsize=(args.desired_size,args.desired_size), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(os.path.join(args.dst_root, f), img)