import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from data_loader import *
from efficientunet import *
from sklearn.metrics import (
    jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score)

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

from utils import create_dir, seeding, make_channel_last
checkpoint_path = "/data01/Sahadev_3rd_paper/Mixer_UNet/files/wid1_CVC.pth"
Batch_size = 16
model_type = 'newidea4'

def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)
def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_dice = dice_score(y_true, y_pred)
    score_jaccard = jaccard_score(y_true, y_pred, average='binary')
    score_f1 = f1_score(y_true, y_pred, average='binary')
    score_recall = recall_score(y_true, y_pred, average='binary')
    score_precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    score_acc = accuracy_score(y_true, y_pred)
    score_fbeta = fbeta_score(y_true, y_pred, beta=1.0, average='binary', zero_division=1)

    return [score_dice,score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]
data_type = 'polyp'
depth = 1
channel_dim = 128
token_dim = 64
width_scale=1
mlp_dim=1
patch_size=14
input_size=1
valid_path = '/data01/skin_lesion/images/valid/images/'
val_image_root = '{}/images/'.format(valid_path)
val_gt_root = '{}/masks/'.format(valid_path)

val_loader = get_loader(val_image_root, val_gt_root, batchsize=Batch_size, trainsize=224, augmentation = False,data_type=data_type)

model = get_efficientmlp(out_channels=1, concat_input=True, pretrained=True,width_scale=width_scale,mlp_dim=mlp_dim,patch_size =patch_size,input_size=input_size,model_typee=model_type,depth=depth,channel_dim=channel_dim,token_dim=token_dim).cuda()
device = torch.device('cuda')
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for i, (x, y) in enumerate(val_loader):
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.float32)

    with torch.no_grad():
        yp, x3_0_up, x3_0_up_se, x2_0_up, x2_0_up_se, x1_0_up, x1_0_up_se = model(x)
        score = calculate_metrics(y, yp)
        metrics_score = list(map(add, metrics_score, score))

print(len(val_loader))
Dice = metrics_score[0] / len(val_loader)
jaccard = metrics_score[1] / len(val_loader)
f1 = metrics_score[2] / len(val_loader)
recall = metrics_score[3] / len(val_loader)
precision = metrics_score[4] / len(val_loader)
acc = metrics_score[5] / len(val_loader)
f2 = metrics_score[6] / len(val_loader)

print(f"Dice coefficient:{Dice} - Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")
