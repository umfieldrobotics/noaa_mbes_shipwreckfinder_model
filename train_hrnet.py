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

def train(train_path, val_path, model_name, save_path, num_epochs, batch_size, lr, using_hillshade, using_inpainted):
    device = torch.device(CUDA if torch.cuda.is_available() else 'cpu')
    
    a = argparse.Namespace(cfg='models/hrnet/config/hrnet_config.py',
                                   local_rank=-1,
                                   opts=[],
                                   seed=304)

    update_config(config, a)
    
    model = models.hrnet.seg_hrnet_ocr.get_seg_model(config).to(device)
    model.to(device)
    
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

    # Training loop
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Epochs"):
        model.train()
        train_loss = []
        train_iou = []
        for data in train_loader:
            image = data['image'].cuda()
            image = torch.hstack([image, image, image])
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
                    image = torch.hstack([image, image, image])
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
                    
                        wandb.log({'VAL__' + data['metadata']['image_name'][j].split('/')[-1]: wandb.Image(np.hstack([image[j].permute(1, 2, 0).cpu().numpy()[:,:,0], 
                                                                                                    np.ones((label[j].shape[0], 5)), 
                                                                                                    label[j], 
                                                                                                    np.ones((label[j].shape[0], 5)), 
                                                                                                    pred[j]]),
                                                                                                    caption="Input | Ground Truth | Prediction")})

        
            model.train()
            avg_val_loss = np.mean(val_loss)
            avg_val_iou = np.mean(val_iou)
            wandb.log({"Validation Loss": avg_val_loss, "Validation IOU": avg_val_iou})

        # Save model every 10 epochs, but make a new one every 1000
        curr_save_path = os.path.join(save_path, f"{model_name}_latest.pt")  # Always the latest checkpoint

        # Save a new model file every 1000 epochs
        if (epoch + 1) % 1000 == 0 or epoch + 1 == num_epochs:
            curr_save_path = os.path.join(save_path, f"{model_name}_e{epoch + 1}.pt")

        # Save every 10 epochs (including newly named files every 1000 epochs)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), curr_save_path)
        


########
# TEST #
########

def test(test_path, weight_path, save_images = False, pred_path=None, batch_size=1):
    # Load the model
    print(weight_path)
    model = Unet(3, 2)
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model.eval()
    # Load all the .npy files within test_path using glob
    test_files = glob.glob(os.path.join(test_path, "*_image.npy"))
    ship_iou_list = []
    terrain_iou_list = []
    f1_list = []
    accuracy_list = []
    precision_list = []
    score_path = os.path.join(pred_path,"scores.txt")

    os.makedirs(pred_path, exist_ok=True)
    if save_images:
        clear_directory(pred_path)
    with open(score_path, "w") as f: # Clear text file
        f.write("")  # Writing an empty string clears the file

    val_dataset = MBESDataset(test_path, byt=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    batch_count = 0
    for data in val_loader:
        image = data['image'].cuda()
        label = data['label'].cuda()

        # pred = model(image[:,0:1,:,:]) # For single channel images
        pred = model(image)
        pred = pred.argmax(dim=1)

        label = label.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        label_flat = label.flatten()
        pred_flat = pred.flatten()

        # Compute metrics if labels are available
        accuracy = accuracy_score(label_flat, pred_flat)  # Flatten to 1D arrays for comparison

        # Calculate scores only on masked values (produces much lower IOU)
        valid_data_mask = (label.flatten() != -1)  # Ignore pixels with no data
        label_flat, pred_flat = label_flat[valid_data_mask], pred_flat[valid_data_mask]

        # Initialize flags
        ship_warned = False
        terrain_warned = False

        # Compute ship IoU and detect warning; don't count IOU if label and pred are both 0's
        with warnings.catch_warnings(record=True) as w_ship:
            warnings.simplefilter("always")
            ship_iou = jaccard_score(label_flat, pred_flat, zero_division="warn", average='binary', pos_label=1)
            ship_warned = any(isinstance(w.message, UndefinedMetricWarning) for w in w_ship)

        # Compute terrain IoU and detect warning
        with warnings.catch_warnings(record=True) as w_terrain:
            warnings.simplefilter("always")
            terrain_iou = jaccard_score(label_flat, pred_flat, zero_division="warn", average='binary', pos_label=0)
            terrain_warned = any(isinstance(w.message, UndefinedMetricWarning) for w in w_terrain)

        f1 = f1_score(label_flat, pred_flat, zero_division=1)
        precision = precision_score(label_flat, pred_flat, zero_division=1)
        tn, fp, fn, tp  = confusion_matrix(label_flat, pred_flat, labels=[0,1]).ravel()

        # iou = jaccard_score(label_flat, pred_flat, zero_division=1, average=None)[2] # I think I need to index at [2] if not masked to 2 classes
        # f1 = f1_score(label_flat, pred_flat, zero_division=1, average=None)[2]
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        f1_list.append(f1)

        if not ship_warned:
            ship_iou_list.append(ship_iou)
        else:
            ship_iou = np.nan
            # print("ship prediction and label are both empty")
        
        if not terrain_warned:
            terrain_iou_list.append(terrain_iou)
        else:
            terrain_iou = np.nan 
        #     print("terrain prediction and label are both empty")

        test_file = test_files[batch_count]
        # np.save(pred_file_path, pred) # Save numpy file 

        # normalized_data = (data - data.min())/(data.max() - data.min())
        # colormap = plt.get_cmap('viridis')  # You can replace 'viridis' with your preferred colormap
        # colored_data = colormap(data)

        if save_images:
            mask = label != -1
            pred[mask == 0] = -1 # set areas with no data to -1 for visualization purposes
            # label[mask == 0] == -1 # optional to show no data mask on label in viz

            img = (255 * image.squeeze().permute(1, 2, 0).cpu().numpy()).astype(np.uint8)

            # Save input image, label, and prediction as a single image
            save_combined_image(image, pred, label, test_file, pred_path, ship_iou, terrain_iou)

            # Save prediction as npy
            # np.save(os.path.join(pred_path, os.path.basename(test_file).replace("_image.npy", "_pred.npy")), pred)
            
            # Save original image as npy
            # np.save(os.path.join(pred_path, os.path.basename(test_file)), img)
            
            # Save original image as png
            # img = Image.fromarray((255*image.squeeze().permute(1,2,0).cpu().numpy()).astype(np.uint8))
            # img.save(os.path.join(pred_path, os.path.basename(test_file).replace("_image.npy", "_image.png"))) # Save image
            
            # Save prediction as png
            # pred_img = Image.fromarray((255*pred[0,...]).astype(np.uint8))
            # pred_img.save(os.path.join(pred_path, os.path.basename(test_file).replace("_image.npy", "_pred.png"))) # Save image
            
            # Save Label as png
            # label = Image.fromarray((255*label.squeeze()).astype(np.uint8))
            # label.save(os.path.join(pred_path, os.path.basename(test_file).replace("_image.npy", "_label.png"))) # Save image
            with open(score_path, "a") as f:
                f.write(f"{os.path.basename(test_file).replace('_image.npy', '')} - Ship IOU: {ship_iou:.4f} - Terrain IOU: {terrain_iou:.4f} - F1 Score: {f1:.4f} - Accuracy: {accuracy:.4f}\n")
    
        batch_count += 1

    # After the loop, calculate the averages
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_f1 = np.mean(f1_list)
    avg_ship_iou = np.mean(ship_iou_list)
    avg_terrain_iou = np.mean(terrain_iou_list)

    # Print the averages
    # print(f"Average Accuracy: {avg_accuracy:.4f}")
    # print(f"Average Precision: {avg_precision:.4f}")
    # print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Ship IoU: {avg_ship_iou:.4f}")
    print(f"Terrain IoU: {avg_terrain_iou:.4f}")

    return avg_ship_iou, avg_terrain_iou


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
    
    EPOCHS = 5000
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    
    wandb.init(
        project="mbes",
        group="hrnet",
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
            save_path="./model_weights", 
            num_epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            lr=LEARNING_RATE,
            using_hillshade=False,
            using_inpainted=True)

    # Numerous test functions for evaluation; change test folder, model and save path as needed
    # test("Test_Ships_Fixed", "Models/cnp_data_5_latest.pt", save_images=True, pred_path="Predictions/synthetic_ships")
    # test("New_Terrain_Test", "Models/cnp_data_5_latest.pt", save_images=True, pred_path="Predictions/synthetic_terrain") # Test_Final is all ships
    # test("New_Test_Combined", "Models/cnp_data_5_latest.pt", save_images=True, pred_path="Predictions/synthetic_combined") # Test_Final is all ships
    # test("Plugin_outputs", "Models/cnp_data_7_e6000.pt", save_images=True, pred_path="Predictions/Plugin_outputs_5")

    # Run all models to compare results in txt file
    # with open("test_log.txt", "w") as f:
    #     for file in sorted(os.listdir("Models")):
    #         if file.endswith(".pt"):
    #                 try:
    #                     with redirect_stdout(f):
    #                         test("New_Test_Combined", os.path.join("Models",file), save_images=False, pred_path="Predictions/synthetic_combined") # Test_Final is all ships
    #                 except Exception as e:
    #                     print("Model:", file)
    #                     print(f"Error occurred")