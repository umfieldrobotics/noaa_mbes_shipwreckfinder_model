import albumentations as A
import argparse
import cv2
import glob 
import matplotlib.pyplot as plt
import numpy as np
import os 
import torch
import torch.nn.functional as F
import wandb
import warnings

from tqdm import tqdm 
from torch.utils.data import DataLoader
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score, f1_score, precision_score

from util.data import MBESDataset
import models.hrnet.seg_hrnet_ocr
from models.hrnet.config import config, update_config
from util.utils import clear_directory, save_combined_image


############################################################################
# Training script for ShipwreckFinder plugin.                              #
# Predicts segmentation masks for shipwrecks on NOAA multibeam sonar data. #
#                                                                          #
# Anja Sheppard, Tyler Smithline                                           #
############################################################################

CUDA = 'cuda:0'

#########
# TRAIN #
#########

def train(train_path, val_path, model_name, model_arch, save_path, num_epochs, batch_size, lr, using_hillshade, using_inpainted):
    device = torch.device(CUDA if torch.cuda.is_available() else 'cpu')
    
    a = argparse.Namespace(cfg='models/hrnet/config/hrnet_config.py',
                                   local_rank=-1,
                                   opts=[],
                                   seed=304)

    update_config(config, a)
    
    model = models.hrnet.seg_hrnet_ocr.get_seg_model(config).to(device)
    model.to(device)
    
    os.makedirs(os.path.join(save_path, model_arch, model_name), exist_ok=True)
    
    # Load dataset and split into train/validation sets
    prob = 0.5
    sec_prob = 0.25
    augmentation = None
    augmentation = A.Compose([
                                # A.CoarseDropout(p=prob),
                                A.HorizontalFlip(p=prob), 
                                A.VerticalFlip(p=prob), 
                                A.SafeRotate(p=prob, border_mode=cv2.BORDER_REFLECT_101),
                                # A.RandomResizedCrop(p=prob, size=(200, 200), interpolation=cv2.INTER_NEAREST),
                                # A.GaussianBlur(p=sec_prob), 
                                # A.ColorJitter(p=sec_prob), 
                                # A.AdditiveNoise(p=sec_prob), 
                                # A.CLAHE(p=sec_prob), 
                                # A.Defocus(p=sec_prob), 
                                # A.RandomShadow(p=prob), 
                                # A.SaltAndPepper(p=sec_prob)
                            ])

    
    aug_multiplier = 0 # number of extra augmentations per image. If 1, dataset will be 2*(original dataset size)
    
    train_dataset = MBESDataset(train_path, byt=False, transform=augmentation, aug_multiplier=aug_multiplier, using_hillshade=using_hillshade, using_inpainted=using_inpainted)
    val_dataset = MBESDataset(val_path, byt=False, using_hillshade=using_hillshade, using_inpainted=using_inpainted)

    print("Train dataset length:", len(train_dataset))

    # Dataloaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    #find how many pixels are 1 vs 0 in labels 
    label_count = 0
    background_count = 0
    total_count = 0
    for data in train_loader:
        label = data['label'].cuda()
        label_count += (label == 1).sum()
        background_count += (label == 0).sum()
    total_count = background_count # made a separate background variable since masked pixels don't count as background
    print(label_count, total_count)

    ratio = label_count / (total_count)
    weight0 = label_count / (total_count - label_count) # weight 0 should be higher because there are less 
    weight1 = 1/weight0
    print("Ratio:", ratio, "Weight0:", weight0, "Weight1:", weight1)
    print("Ones Count", label_count, "Zeros count:", total_count-label_count)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([weight0, weight1]).cuda(), ignore_index=-1)
    
    best_iou = 0

    # Training loop
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Epochs"):
        model.train()
        train_loss = []
        train_iou = []
        for data in train_loader:
            image = data['image'].cuda()
            label = data['label'].cuda()

            optim.zero_grad()

            pred = model(image)[0]
            pred = F.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=True)
            loss = ce_loss(pred.float(), label)
            pred = pred.argmax(dim=1)

            label = label.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            valid_data_mask = (label.flatten() != -1)  # Ignore invalid pixels
            label_flat, pred_flat = label.flatten()[valid_data_mask], pred.flatten()[valid_data_mask]
            iou = jaccard_score(label_flat, pred_flat, zero_division=1)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_iou.append(iou)

            # For validating training data
            # if epoch % 50 == 0:
            #     wandb.log({"Train Image": wandb.Image(image[0].permute(1, 2, 0).cpu().numpy(), caption="Train Input")})
            #     wandb.log({"Train Label": wandb.Image(label[0].cpu().numpy(), caption="Train Label")})
        
        avg_train_iou = np.mean(train_iou)
        wandb.log({"Epoch": epoch, "Train Loss": np.mean(train_loss), "Train IOU": avg_train_iou})

        # Validation loop
        model.eval()
        val_loss = []
        val_iou = []
        # preds = []
        if epoch % 50 == 0:
            with torch.no_grad():
                for idx, data in enumerate(val_loader):
                    image = data['image'].cuda()
                    label = data['label'].cuda()
                    # image = (image - image.min(dim=(1,2,3)))/(image.max(dim=(1,2,3)) - image.min(dim=(1,2,3)))
                    pred = model(image)[0]
                    pred = F.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=True)
                    loss = ce_loss(pred, label.long())
                    pred = pred.argmax(dim=1)
                    # preds.append(pred)
                    label = label.cpu().detach().numpy()
                    pred = pred.cpu().detach().numpy()
                    valid_data_mask = (label.flatten() != -1)  # Ignore 255
                    label_flat, pred_flat = label.flatten()[valid_data_mask], pred.flatten()[valid_data_mask]
                    iou = jaccard_score(label_flat, pred_flat, zero_division=1)

                    val_loss.append(loss.item())
                    val_iou.append(iou)

                    for j in range(len(label)):
                        pred[j][label[j] == -1] = -1
                    
                        wandb.log({"Val_Epoch": epoch, 'VAL__' + data['metadata']['image_name'][j].split('/')[-1]: wandb.Image(np.hstack([image[j].permute(1, 2, 0).cpu().numpy()[:,:,0], 
                                                                                                    np.ones((label[j].shape[0], 5)), 
                                                                                                    label[j], 
                                                                                                    np.ones((label[j].shape[0], 5)), 
                                                                                                    pred[j]]),
                                                                                                    caption="Input | Ground Truth | Prediction")})

        
            model.train()
            avg_val_loss = np.mean(val_loss)
            avg_val_iou = np.mean(val_iou)
            wandb.log({"Validation Loss": avg_val_loss, "Validation IOU": avg_val_iou})

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

def test(test_path, weight_path, model_name, model_arch, using_hillshade, using_inpainted, save_images=False, pred_path=None, batch_size=1):
    print("Testing " + weight_path)
    
    # Load the model    
    a = argparse.Namespace(cfg='models/hrnet/config/hrnet_config.py',
                                   local_rank=-1,
                                   opts=[],
                                   seed=304)

    update_config(config, a)

    model = models.hrnet.seg_hrnet_ocr.get_seg_model(config)
    model.load_state_dict(torch.load(weight_path))
    model.cuda()

    model.eval()

    test_dataset = MBESDataset(test_path, byt=False, using_hillshade=using_hillshade, using_inpainted=using_inpainted)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Set up metric tracking + output files
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
            image = data['image'].cuda()
            label = data['label'].cuda()

            pred = model(image)[0]
            pred = F.interpolate(pred, size=label.shape[1:], mode='bilinear', align_corners=True)
            pred = pred.argmax(dim=1)

            label = label.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            
            # Remove small contours
            min_area = 150  # adjust this value as needed
            cleaned_pred = []

            for p in pred:
                p = p.astype(np.uint8)
                contours, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cleaned = np.zeros_like(p)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area >= min_area:
                        cv2.drawContours(cleaned, [cnt], -1, 1, thickness=cv2.FILLED)
                cleaned_pred.append(cleaned.astype(np.int32))

            pred = np.stack(cleaned_pred, axis=0)
            
            valid_data_mask = (label.flatten() != -1)  # Ignore invalid pixels
            label_flat = label.flatten()[valid_data_mask]
            pred_flat = pred.flatten()[valid_data_mask]

            tn, fp, fn, tp  = confusion_matrix(label_flat, pred_flat, labels=[0,1]).ravel()
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_tp += tp
            
            iou_ship = tp / (fp + tp + fn)

            if save_images:
                pred[label == -1] = -1
                # mask = (label != -1)
                # pred[mask == 0] = -1 # set areas with no data to -1 for visualization purposes
                # label[mask == 0] == -1 # optional to show no data mask on label in viz

                # Save input image, label, and prediction as a single image
                combined_img = save_combined_image(image[:,0,:], pred, label, data['metadata']['image_name'][0], pred_path, iou_ship)
                
                wandb.log({'TEST__' + data['metadata']['image_name'][0].split('/')[-1]: wandb.Image(combined_img,
                                                                            caption="Input | Ground Truth | Prediction")})
            
        # After the loop, calculate the averages
        accuracy = total_tp / (total_tp + total_fp + total_fn)
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
        iou_ship = total_tp / (total_fp + total_tp + total_fn)
        # for terrain, ship tn = terrain tp, ship tp = terrain tn, ship fp = terrain fn, ship fn = terrain fp
        iou_background = total_tn / (total_fn + total_tn + total_fp)

        # Print the averages
        print(f"Average Accuracy: {accuracy:.4f}")
        print(f"Average Precision: {precision:.4f}")
        print(f"Average F1 Score: {f1:.4f}")
        print(f"Average Recall: {recall:.4f}")
        print(f"Average Ship IoU: {iou_ship:.4f}")
        print(f"Average Background IoU: {iou_background:.4f}")
        
        wandb.log({"Average Accuracy": accuracy, "Average Precision": precision, "Average F1 Score": f1, "Average Recall": recall, "Average Ship IoU": iou_ship, "Average Background IoU": iou_background})


########
# EVAL #
########

def evaluate(batch_size):
    model = Unet(3, 2)
    model.cuda()

    # Load dataset and split into train/validation sets
    # dataset = MBESDataset("/mnt/syn/advaiths/datasets/mbes_data/real_data/Train_Final_filtered", byt=False)
    dataset = MBESDataset("Combined_Ships_Fixed", byt=False)

    # Dataloaders for train and validation sets
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    #find how many pixels are 1 vs 0 in labels 
    label_count = 0
    total_count = 0
    mean_depths = []
    for data in train_loader:
        label = data['label'].cuda()
        image = data['image'].cuda()
        print("Data shape", image.shape, label.shape)
        masked_image = label*image
        # if(label_count == 0):
        #     plt.imshow(masked_image.cpu()[0,0,...], cmap='viridis')
        #     plt.colorbar()
        #     plt.show()
        mean_depths.append(masked_image.sum().item() / label.sum().item())
        label_count += label.sum()
        total_count += label.numel()
    print(label_count, total_count)

    ratio = label_count / (total_count)
    weight1 = label_count / (total_count - label_count)
    weight0 = 1/weight1
    print("Ratio:", ratio.item(), "Weight0:", weight0.item(), "Weight1:", weight1.item())
    print("Ones Count", label_count.item(), "Zeros count:", (total_count-label_count).item())
    # print("Means", mean_depths)

    plt.hist(mean_depths, bins = 20, edgecolor='black')
    plt.xlabel('Depth')
    plt.ylabel('Frequency')
    plt.title('Distribution of Shipwreck Mean Depths')
    plt.show()



if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    EPOCHS = 12000
    BATCH_SIZE = 8
    LEARNING_RATE = 7e-4
    USING_HILLSHADE = False
    USING_INPAINTED = True
    
    wandb.init(
        project="mbes",
        group="hrnet",
        # id="ap7zd22j",
        # resume="must",
        config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "model_type": "hrnet"
               }
    )
    
    train(train_path="/frog-drive/noaa_multibeam/Synthetic_Dataset/Train_With_Synthetic",
            val_path="/frog-drive/noaa_multibeam/Synthetic_Dataset/Val_With_Synthetic",
            model_name=wandb.run.name,
            model_arch=wandb.run.group,
            save_path="./model_weights", 
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE, 
            lr=LEARNING_RATE,
            using_hillshade=USING_HILLSHADE,
            using_inpainted=USING_INPAINTED)

    test(test_path="/frog-drive/noaa_multibeam/Synthetic_Dataset/Test_With_Terrain",
         weight_path=os.path.join("model_weights", wandb.run.group, wandb.run.name, wandb.run.name+"_best.pt"),
         model_name=wandb.run.name,
         model_arch=wandb.run.group,
         using_hillshade=USING_HILLSHADE, 
         using_inpainted=USING_INPAINTED, 
         save_images=True, 
         pred_path="model_outputs")