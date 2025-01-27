import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets

import args
import utils
from networks import LossNetwork, StyleBankNet

"""********Important*******"""
args.continue_training = False # change to your setting
"""************************"""

device = args.device
"""
Load Dataset
"""
content_dataset = datasets.ImageFolder(root=args.CONTENT_IMG_DIR, transform=utils.content_img_transform)
content_dataloader = torch.utils.data.DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

style_dataset = datasets.ImageFolder(root=args.STYLE_IMG_DIR, transform=utils.style_img_transform)
style_dataset = torch.cat([img[0].unsqueeze(0) for img in style_dataset], dim=0)
style_dataset = style_dataset.to(device)
"""
Define Model and Loss Network (vgg16)
"""
model = StyleBankNet(len(style_dataset)).to(device)

if args.continue_training:
    if os.path.exists(args.GLOBAL_STEP_PATH):
        with open(args.GLOBAL_STEP_PATH, 'r') as f:
            global_step = int(f.read())
    else:
        raise Exception('cannot find global step file')
    if os.path.exists(args.MODEL_WEIGHT_PATH):
        model.load_state_dict(torch.load(args.MODEL_WEIGHT_PATH))
    else:
        raise Exception('cannot find model weights')
else:
    if not os.path.exists(args.MODEL_WEIGHT_DIR):
        os.mkdir(args.MODEL_WEIGHT_DIR)
    if not os.path.exists(args.BANK_WEIGHT_DIR):
        os.mkdir(args.BANK_WEIGHT_DIR)
    global_step = 0
        
optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer_ae = optim.Adam(model.parameters(), lr=args.lr)
loss_network = LossNetwork().to(device)
"""
Training
"""

# [0, 1, 2, ..., N]
style_id = list(range(len(style_dataset)))
style_id_idx = 0
style_id_seg = []
for i in range(0, len(style_dataset), args.batch_size):
    style_id_seg.append(style_id[i:i+args.batch_size])
    
s_sum = 0 # sum of style loss
c_sum = 0 # sum of content loss
l_sum = 0 # sum of style+content loss
r_sum = 0 # sum of reconstruction loss
tv_sum = 0 # sum of tv loss

while global_step <= args.MAX_ITERATION:
    for i, data in enumerate(content_dataloader):
        global_step += 1
        data = data[0].to(device)
        batch_size = data.shape[0]
        if global_step % (args.T+1) != 0:
            style_id_idx += 1
            sid = utils.get_sid_batch(style_id_seg[style_id_idx % len(style_id_seg)], batch_size)
            
            optimizer.zero_grad()
            output_image = model(data, sid)
            content_score, style_score = loss_network(output_image, data, style_dataset[sid])
            content_loss = args.CONTENT_WEIGHT * content_score
            style_loss = args.STYLE_WEIGHT * style_score
            
            diff_i = torch.sum(torch.abs(output_image[:, :, :, 1:] - output_image[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(output_image[:, :, 1:, :] - output_image[:, :, :-1, :]))
            tv_loss = args.REG_WEIGHT*(diff_i + diff_j)
            
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            l_sum += total_loss.item()
            s_sum += style_loss.item()
            c_sum += content_loss.item()
            tv_sum += tv_loss.item()

        if global_step % (args.T+1) == 0:
            optimizer_ae.zero_grad()
            output_image = model(data)
            loss = F.mse_loss(output_image, data)
            loss.backward()
            optimizer_ae.step()
            r_sum += loss.item()
            
        if global_step % 100 == 0:
            print('.', end='')
            
        if global_step % args.LOG_ITER == 0:
            print("gs: {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(global_step / args.K, time.strftime("%H:%M:%S"), l_sum / 666, c_sum / 666, s_sum / 666, tv_sum / 666, r_sum / 333))
            r_sum = 0
            s_sum = 0
            c_sum = 0
            l_sum = 0
            tv_sum = 0
            # save whole model (including stylebank)
            torch.save(model.state_dict(), args.MODEL_WEIGHT_PATH)
            # save seperate part
            with open(args.GLOBAL_STEP_PATH, 'w') as f:
                f.write(str(global_step))
            torch.save(model.encoder_net.state_dict(), args.ENCODER_WEIGHT_PATH)
            torch.save(model.decoder_net.state_dict(), args.DECODER_WEIGHT_PATH)
            for i in range(len(style_dataset)):
                torch.save(model.style_bank[i].state_dict(), args.BANK_WEIGHT_PATH.format(i))
            
        if global_step % args.ADJUST_LR_ITER == 0:
            lr_step = global_step / args.ADJUST_LR_ITER
            utils.adjust_learning_rate(optimizer, lr_step)
            new_lr = utils.adjust_learning_rate(optimizer_ae, lr_step)
            
            print("learning rate decay:", new_lr)