import numpy as np
import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from util.utils import normalize_nonzero


class MBESDataset(Dataset):
    def __init__(self, root_dir, transform=None, byt=False, aug_multiplier=0):
        self.root_dir = root_dir
        self.transform = transform
        self.byt = byt
        self.aug_multiplier = aug_multiplier  # Number of additional augmented samples per image
        self.img_size = 200

        self.file_list = [file_name for file_name in os.listdir(root_dir) if "_image.npy" in file_name]
        self.resize = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)  # Resize to 501x501

        self.expanded_file_list = [(file_name, i) for file_name in self.file_list for i in range(aug_multiplier + 1)]

    def __len__(self):
        return len(self.expanded_file_list)
 
    def __getitem__(self, idx):
        file_name, aug_idx = self.expanded_file_list[idx]
        image_name = os.path.join(self.root_dir, file_name)
        label_name = image_name.replace("_image.npy", "_label.npy")

        # Image is already a float
        image = torch.from_numpy(np.load(image_name))

        if os.path.exists(label_name):
            label = torch.from_numpy(np.load(label_name)) > 0
        else:
            label = torch.zeros((self.img_size, self.img_size), dtype=torch.long)  # Assign all zeros if no label

        
        # Resize image and label for transformationsdef clear_directory(directory_path):
        image = image.squeeze()
        if len(image.shape) < 3:
            image = image.unsqueeze(0)  # Ensure shape is (1, H, W) for single channel image 
            image = torch.cat([image, image, image], dim=0)

        # Normalize image to [0, 1], might improve results but we may want to maintain depth values
        image = self.resize(image)  # Resize
        mask = (image[0] == 0).int()

        # Testing, looks fine here 
        depth_grid = image.permute(1,2,0).cpu().numpy()

        image = normalize_nonzero(image)

        label = label.unsqueeze(0).float()  # Ensure shape is (1, H, W) for label
        label = self.resize(label).squeeze(0).long()  # Resize and remove singleton dimension
        image = image.permute(1,2,0) # permute for transformations

        # print(file_name, image.mean())

        # Convert to numpy for transformations
        image, label, mask = image.numpy().astype(np.float32), label.numpy(), mask.numpy()

        # label[] = -1
        masks = [(mask * 255).astype(np.int32), label.astype(np.int32)]

        # Apply transforms if provided
        if self.transform: 
            transformed = self.transform(image=image, masks=masks)
            image, masks = transformed["image"], transformed["masks"]

        transformed_mask = torch.tensor(masks[0], dtype = torch.long)
        label = torch.tensor(masks[1], dtype = torch.long)
        # label = masks[1]
        label[transformed_mask == 255] = -1

        # Save images to confirm augmentations
        # np.save(os.path.join('QGIS_Chunks', os.path.basename(file_name)), image) # Save image
        # save_plot(os.path.join('Augmented_Ships', os.path.basename(file_name.replace("_image.npy", "_normalized.png"))), image)
        # if self.transform:
        #     img = Image.fromarray((255*image).astype(np.uint8))
        #     img.save(os.path.join('Augmented_Ships', os.path.basename(file_name.replace("_image.npy", "_augmented.png")))) # Save image
        #     lab = Image.fromarray((127.5*label.numpy()+127.5).astype(np.uint8)) # scaled for viz purposes 
        #     lab.save(os.path.join('Augmented_Ships', os.path.basename(file_name.replace("_image.npy", "_augmented_label.png")))) # Save image

        return {'image': torch.tensor(image).permute(2, 0, 1).float(), 'label': label, 'metadata': {"image_name": image_name, "label_name": label_name}}