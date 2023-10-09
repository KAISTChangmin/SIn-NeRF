from metrics.clip_metrics import *
import imageio.v2 as imageio
from tqdm import tqdm
import os

NERF = "farm-small"
IN2N = "farm_desert"
text_0 = "photograph of a farm"
text_1 = "photograph of a namibian desert”"



paths0 = sorted(os.listdir(f"renders/images/{NERF}"))
paths1 = paths0[1:] + paths0[:1]

C = ClipSimilarity()

text_features_0 = C.encode_text(text_0)
text_features_1 = C.encode_text(text_1)

results = {'sim_direction': [], 'sim_consistency': []}

for fname0, fname1 in tqdm(zip(paths0, paths1)):
    image_0 = torch.FloatTensor(imageio.imread(f"renders/images/{NERF}/{fname0}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    image_1 = torch.FloatTensor(imageio.imread(f"renders/images/{IN2N}/{fname0}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    image_2 = torch.FloatTensor(imageio.imread(f"renders/images/{NERF}/{fname1}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    image_3 = torch.FloatTensor(imageio.imread(f"renders/images/{IN2N}/{fname1}")).unsqueeze(dim=0).permute(0, 3, 1, 2) / 255
    
    image_features_0 = C.encode_image(image_0)
    image_features_1 = C.encode_image(image_1)
    image_features_2 = C.encode_image(image_2)
    image_features_3 = C.encode_image(image_3)

    sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
    sim_consistency = F.cosine_similarity(image_features_1 - image_features_0, image_features_3 - image_features_2)
    
    results['sim_direction'].append(sim_direction)
    results['sim_consistency'].append(sim_consistency)

print(torch.cat(results['sim_direction'], 0).mean())
print(torch.cat(results['sim_consistency'], 0).mean())