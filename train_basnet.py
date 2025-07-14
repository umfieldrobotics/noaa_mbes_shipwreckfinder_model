import albumentations as A
import cv2
import glob
import numpy as np
import os
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score, f1_score, precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import warnings

from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.BASNet.model import BASNet
import models.BASNet.pytorch_ssim as pytorch_ssim
import models.BASNet.pytorch_iou as pytorch_iou

from util.data import MBESDataset
from util.utils import clear_directory, save_combined_image


############################################################################
# Training script for ShipwreckFinder plugin.                              #
# Predicts segmentation masks for shipwrecks on NOAA multibeam sonar data. #
# Adapted from BASNet (https://github.com/xuebinqin/BASNet)                #
#                                                                          #
# Anja Sheppard, Tyler Smithline                                           #
############################################################################

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred, target, ignore_index=None):
    if ignore_index is not None:
        # mask of valid pixels: shape [B, C, H, W]
        mask = (target != ignore_index)
        valid_pixels = mask.sum()

        if valid_pixels.item() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # For BCE: flatten and select only valid pixels
        pred_valid = pred[mask]
        target_valid = target[mask]

        # sanity check: target_valid should be in [0,1]
        if (target_valid < 0).any() or (target_valid > 1).any():
            raise ValueError("Target values outside [0,1] in valid pixels")

        bce_out = F.binary_cross_entropy(pred_valid, target_valid, reduction='mean')

        # For SSIM and IoU, still use full tensors (they expect 4D)
        ssim_out = 1 - ssim_loss(pred, target)
        iou_out = iou_loss(pred, target)
    else:
        bce_out = F.binary_cross_entropy(pred, target)
        ssim_out = 1 - ssim_loss(pred, target)
        iou_out = iou_loss(pred, target)

    loss = bce_out + ssim_out + iou_out
    return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v, ignore_index=None):
    loss0 = bce_ssim_loss(d0, labels_v, ignore_index)
    loss1 = bce_ssim_loss(d1, labels_v, ignore_index)
    loss2 = bce_ssim_loss(d2, labels_v, ignore_index)
    loss3 = bce_ssim_loss(d3, labels_v, ignore_index)
    loss4 = bce_ssim_loss(d4, labels_v, ignore_index)
    loss5 = bce_ssim_loss(d5, labels_v, ignore_index)
    loss6 = bce_ssim_loss(d6, labels_v, ignore_index)
    loss7 = bce_ssim_loss(d7, labels_v, ignore_index)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

    # print("l0: %.6f, l1: %.6f, l2: %.6f, l3: %.6f, l4: %.6f, l5: %.6f, l6: %.6f, l7: %.6f" % (
    #     loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item(), loss7.item()))

    return loss0, loss

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn


#########
# TRAIN #
#########

def train(train_path, val_path, model_name, model_arch, save_path, num_epochs, batch_size, lr, threshold, using_hillshade, using_inpainted):

    model = BASNet(3, 1)
    model.cuda()
    
    os.makedirs(os.path.join(save_path, model_arch, model_name), exist_ok=True)

    # Load dataset and split into train/validation sets
    prob = 0.5
    augmentation = None
    augmentation = A.Compose([
                                A.HorizontalFlip(p=prob), 
                                A.VerticalFlip(p=prob), 
                                A.SafeRotate(p=prob, border_mode=cv2.BORDER_REFLECT_101),
                            ])

    
    aug_multiplier = 0 # number of extra augmentations per image. If 1, dataset will be 2*(original dataset size)
    
    train_dataset = MBESDataset(train_path, byt=False, transform=augmentation, aug_multiplier=aug_multiplier, using_hillshade=using_hillshade, using_inpainted=using_inpainted, resize_to_div_16=True)
    val_dataset = MBESDataset(val_path, byt=False, using_hillshade=using_hillshade, using_inpainted=using_inpainted, resize_to_div_16=True)

    print("Train dataset length:", len(train_dataset))

    # Dataloaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    best_iou = 0

    # Training loop
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Epochs"):
        model.train()
        train_loss = []
        train_tar_loss = []
        train_iou = []
        for data in train_loader:
            image = data['image'].type(torch.FloatTensor)
            image = torch.hstack([image, image, image])
            label = data['label'].unsqueeze(1).type(torch.FloatTensor)

            image_v, label_v = Variable(image, requires_grad=False).cuda(), Variable(label, requires_grad=False).cuda()

            optimizer.zero_grad()
            
            d0, d1, d2, d3, d4, d5, d6, d7 = model(image_v)
            loss_tar, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, label_v, ignore_index=-1)
            
            loss.backward()
            optimizer.step()
            
            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            pred = (pred >= threshold).unsqueeze(1).type(torch.IntTensor)

            label = label.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            valid_data_mask = (label.flatten() != -1)  # Ignore invalid pixels
            label_flat, pred_flat = label.flatten()[valid_data_mask], pred.flatten()[valid_data_mask]
            iou = jaccard_score(label_flat, pred_flat, zero_division=1)

            train_loss.append(loss.item())
            train_tar_loss.append(loss_tar.item())
            train_iou.append(iou)
        
        avg_train_iou = np.mean(train_iou)
        wandb.log({"Epoch": epoch, "Train Loss": np.mean(train_loss), "Train Tar Loss": np.mean(train_tar_loss), "Train IOU": avg_train_iou})

        # Validation loop
        model.eval()
        val_loss = []
        val_tar_loss = []
        val_iou = []
        if epoch % 25 == 0:
            for idx, data in enumerate(val_loader):
                image = data['image'].type(torch.FloatTensor)
                image = torch.hstack([image, image, image])
                label = data['label'].unsqueeze(1).type(torch.FloatTensor)
                
                image_v, label_v = Variable(image, requires_grad=False).cuda(), Variable(label, requires_grad=False).cuda()

                optimizer.zero_grad()
                
                d0, d1, d2, d3, d4, d5, d6, d7 = model(image_v)
                loss_tar, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, label_v, ignore_index=-1)
                
                loss.backward()
                
                pred = d1[:,0,:,:]
                pred = normPRED(pred)
                pred = (pred >= threshold).unsqueeze(1).type(torch.IntTensor)

                label = label.cpu().detach().numpy()
                pred = pred.cpu().detach().numpy()
                valid_data_mask = (label.flatten() != -1)  # Ignore 255
                label_flat, pred_flat = label.flatten()[valid_data_mask], pred.flatten()[valid_data_mask]
                iou = jaccard_score(label_flat, pred_flat, zero_division=1)

                val_loss.append(loss.item())
                val_tar_loss.append(loss_tar.item())
                val_iou.append(iou)
                
                for j in range(len(label)):
                    pred[label == -1] = -1
                    wandb.log({"Val_Epoch": epoch, 'VAL__' + data['metadata']['image_name'][j].split('/')[-1]: wandb.Image(np.hstack([image[j].permute(1, 2, 0).cpu().numpy()[:,:,0], 
                                                                                                np.ones((label[j].shape[1], 5)), 
                                                                                                label[j][0], 
                                                                                                np.ones((label[j].shape[1], 5)), 
                                                                                                pred[j][0]]), caption="Input | Ground Truth | Prediction")})

        
            model.train()
            avg_val_loss = np.mean(val_loss)
            avg_val_tar_loss = np.mean(val_tar_loss)
            avg_val_iou = np.mean(val_iou)
            wandb.log({"Validation Loss": avg_val_loss, "Validation Loss Tar": avg_val_tar_loss, "Validation IOU": avg_val_iou})

        # Save model (best, latest, and every 1000 epochs)
        curr_save_path = os.path.join(save_path, model_arch, model_name, model_name)

        # Save every 10 epochs (including newly named files every 1000 epochs)
        if (epoch + 1) % 10 == 0:
            if (epoch + 1) % 1000 == 0:
                torch.save(model.state_dict(), curr_save_path + f"e{epoch + 1}.pt")
            else:
                torch.save(model.state_dict(), curr_save_path + "_latest.pt")
        
        if train_iou[-1] > best_iou:
            best_iou = train_iou[-1]
            torch.save(model.state_dict(), curr_save_path + "_best.pt")
        


########
# TEST #
########

def test(test_path, weight_path, model_name, model_arch, threshold, using_hillshade, using_inpainted, save_images=False, pred_path=None, batch_size=1):
    print("Testing " + weight_path)
    
    # Load the model
    model = BASNet(3, 1)
    model.load_state_dict(torch.load(weight_path))
    model.cuda()

    model.eval()

    test_dataset = MBESDataset(test_path, byt=False, using_hillshade=using_hillshade, using_inpainted=using_inpainted, resize_to_div_16=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Set up metric tracking + output files
    ship_iou_list = []
    f1_list = []
    accuracy_list = []
    precision_list = []
    total_tn = 0
    total_tp = 0
    total_fn = 0
    total_fp = 0

    pred_path = os.path.join(pred_path, model_arch, model_name)
    os.makedirs(pred_path, exist_ok=True)
    if save_images:
        clear_directory(pred_path)

    with torch.no_grad():
        for data in test_loader:
            image = data['image'].type(torch.FloatTensor)
            image = torch.hstack([image, image, image])
            label = data['label'].unsqueeze(1).type(torch.FloatTensor)
            
            image_v = Variable(image, requires_grad=False).cuda()

            _, d1, _, _, _, _, _, _ = model(image_v)

            pred = d1[:,0,:,:]
            pred = normPRED(pred)
            pred = (pred >= threshold).unsqueeze(1).type(torch.IntTensor)

            label = label.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            
            valid_data_mask = (label.flatten() != -1)  # Ignore invalid pixels
            label_flat = label.flatten()[valid_data_mask]
            pred_flat = pred.flatten()[valid_data_mask]
            
            # Initialize flags
            ship_warned = False

            # Compute metrics if labels are available
            accuracy = accuracy_score(label_flat, pred_flat)  # Flatten to 1D arrays for comparison

            # Compute ship IoU and detect warning; don't count IOU if label and pred are both 0's
            with warnings.catch_warnings(record=True) as w_ship:
                warnings.simplefilter("always")
                ship_iou = jaccard_score(label_flat, pred_flat, zero_division="warn", average='binary', pos_label=1)
                ship_warned = any(isinstance(w.message, UndefinedMetricWarning) for w in w_ship)

            f1 = f1_score(label_flat, pred_flat, zero_division=1)
            precision = precision_score(label_flat, pred_flat, zero_division=1)
            tn, fp, fn, tp  = confusion_matrix(label_flat, pred_flat, labels=[0,1]).ravel()
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_tp += tp

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            f1_list.append(f1)

            if not ship_warned:
                ship_iou_list.append(ship_iou)
            else:
                ship_iou = np.nan
                print("ship prediction and label are both empty")

            if save_images:
                mask = (label != -1)
                pred[mask == 0] = -1 # set areas with no data to -1 for visualization purposes
                label[mask == 0] == -1 # optional to show no data mask on label in viz

                # Save input image, label, and prediction as a single image
                combined_img = save_combined_image(image[0,0,:,:], pred[0], label[0], data['metadata']['image_name'][0], pred_path, ship_iou)
                
                wandb.log({'TEST__' + data['metadata']['image_name'][0].split('/')[-1]: wandb.Image(combined_img,
                                                                            caption="Input | Ground Truth | Prediction")})
        
    # After the loop, calculate the averages
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_f1 = np.mean(f1_list)
    avg_ship_iou = np.mean(ship_iou_list)

    # Print the averages
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Ship IoU: {avg_ship_iou:.4f}")
    
    wandb.log({"Average Accuracy": avg_accuracy, "Average Precision": avg_precision, "Average F1 Score": avg_f1, "Average Ship IoU": avg_ship_iou})


if __name__ == "__main__":
    EPOCHS = 7000
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4
    THRESHOLD = 0.5
    USING_HILLSHADE = False
    USING_INPAINTED = True
    
    wandb.init(
        project="mbes",
        group="basnet",
        config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "threshold": THRESHOLD,
                "model_type": "unet"
               }
    )
    
    train(train_path="/frog-drive/noaa_multibeam/Synthetic_Dataset/Train_With_Synthetic_Aux",
            val_path="/frog-drive/noaa_multibeam/Synthetic_Dataset/Val_With_Synthetic_Aux",
            model_name=wandb.run.name,
            model_arch=wandb.run.group,
            save_path="./model_weights", 
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE, 
            lr=LEARNING_RATE,
            threshold=THRESHOLD,
            using_hillshade=USING_HILLSHADE,
            using_inpainted=USING_INPAINTED)

    test(test_path="/frog-drive/noaa_multibeam/Synthetic_Dataset/Test_Aux",
         weight_path=os.path.join("model_weights", wandb.run.group, wandb.run.name, wandb.run.name+"_best.pt"),
         model_name=wandb.run.name,
         model_arch=wandb.run.group,
         threshold=THRESHOLD,
         using_hillshade=USING_HILLSHADE, 
         using_inpainted=USING_INPAINTED, 
         save_images=True, 
         pred_path="model_outputs")