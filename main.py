import torch
from torch.autograd import Variable
import os
import time
import argparse
from datetime import datetime
import torch.nn.functional as F
import numpy as np
from losses import *
from utils import *
from efficientunet import *
from data_loader import *
#from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Input some variables.')
parser.add_argument('--lr', default=0.0001, type=float,
                        help='define learning rate',required=False)
parser.add_argument('--name', type=str, default='model',
                        help='define a model name',required=False)
parser.add_argument('--data_type', type=str, default='polyp',
                        help='define a data type',required=False)
parser.add_argument('--train_path', type=str, default='/data01/skin_lesion/images/train/images/',
                        help='define a data path',required=False)
parser.add_argument('--valid_path', type=str, default='/data01/skin_lesion/images/valid/images/',
                        help='define a data path',required=False)
parser.add_argument('--width_scale', type=int, default=1,
                        help='Give width of the network ',required=False)
parser.add_argument('--mlp_dim', type=int, default=1,
                        help='Give mlp dimension',required=False)
parser.add_argument('--depth', type=int, default=8,
                        help='Give mlp depth',required=False)
parser.add_argument('--channel_dim', type=int, default=128,
                        help='Give mlp channeld imension',required=False)
parser.add_argument('--token_dim', type=int, default=64,
                        help='Give mlp token',required=False)
parser.add_argument('--patch_size', type=int, default=14,
                        help='define a patch size',required=False)
parser.add_argument('--input_size', type=int, default=1,
                        help='Give an input resolution',required=False)
parser.add_argument('--batch_size', type=int, default=16,
                        help='Give an batch size',required=False)

args = parser.parse_args()
model_name=args.name
data_type = args.data_type
lr=args.lr
Batch_size = args.batch_size
input_size = args.input_size
""" Seeding """
seeding(42)

""" Directories """
create_dir("files")

""" Training logfile """
train_log_path = "files/"+model_name+".txt"
if os.path.exists(train_log_path):
    print("Log file exists")
else:
    train_log = open("files/"+model_name+".txt", "w")
    train_log.write("\n")
    train_log.close()

#writer = SummaryWriter(comment=f'Model_name_{args.name}')
#global_step=0

train_path = args.train_path
image_root = '{}/images/'.format(train_path)
gt_root = '{}/masks/'.format(train_path)
valid_path = args.valid_path
val_image_root = '{}/images/'.format(valid_path)
val_gt_root = '{}/masks/'.format(valid_path)
checkpoint_path = "files/"+model_name+".pth"


train_loader = get_loader(image_root, gt_root, batchsize=Batch_size, trainsize=224*args.input_size, augmentation=True,data_type=data_type)
total_step = len(train_loader)
val_loader = get_loader(val_image_root, val_gt_root, batchsize=Batch_size, trainsize=224*args.input_size, augmentation=False,data_type=data_type)

model = get_efficientmlp(out_channels=1, concat_input=True, pretrained=True,width_scale=args.width_scale,mlp_dim=args.mlp_dim,patch_size =args.patch_size,input_size=args.input_size,depth=args.depth,channel_dim=args.channel_dim,token_dim=args.token_dim).cuda()
#print(model)

device = torch.device('cuda')
model = model.to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

loss_fn = DiceBCELoss()

best_valid_loss = float('inf')

data_str = f"Hyperparameters:\nImage Size: {224*input_size}\nBatch Size: {Batch_size}\nLR: {lr}\nEpochs: {200}\n"
data_str += f"Optimizer: Adam\nLoss: {loss_fn}\n"
print_and_save(train_log_path, data_str)

for epoch in range(200):
    start_time = time.time()
    epoch_loss = 0

    model.train()
    for i, (x, y) in enumerate(train_loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            yp,x3_0_up,x3_0_up_se,x2_0_up,x2_0_up_se,x1_0_up,x1_0_up_se = model(x)
            loss = loss_fn(yp, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #Image.imsave("./result/atten_map/" + str(epoch) + ".png", attention);
            # writer.add_scalar('Loss/train',loss.item(),global_step)
            # writer.add_images('Training_images', x, global_step)
            # writer.add_images('train_masks/true', y, global_step)
            # writer.add_images('train_masks/pred', torch.sigmoid(yp) > 0.5, global_step)

            #train_validate(y,yp,epoch,train_loader)

    epoch_loss = epoch_loss/len(train_loader)

    val_epoch_loss = 0
    #global_step +=1

    model.eval()
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(val_loader):
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            yp,x3_0_up,x3_0_up_se,x2_0_up,x2_0_up_se,x1_0_up,x1_0_up_se = model(x_val)

            val_loss = loss_fn(yp, y_val)
            val_epoch_loss += val_loss.item()
            # writer.add_scalar('Loss/valid', val_loss.item(), global_step)
            # writer.add_images('Validation_images',x_val,global_step)
            # writer.add_images('masks/true',y_val,global_step)
            # writer.add_images('masks/pred',torch.sigmoid(yp)>0.5,global_step)

    val_epoch_loss = val_epoch_loss / len(val_loader)

    scheduler.step(val_epoch_loss)

    if val_epoch_loss < best_valid_loss:
        best_valid_loss = val_epoch_loss
        torch.save(model.state_dict(), checkpoint_path)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    data_str = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
    data_str += f'\tTrain Loss: {epoch_loss:.3f}\n'
    data_str += f'\t Val. Loss: {val_epoch_loss:.3f}\n'
    print_and_save(train_log_path, data_str)
#writer.close()
