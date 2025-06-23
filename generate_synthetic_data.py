import os
import numpy as np
import random
import cv2
from tqdm import tqdm
from util.utils import clear_directory, copy_files
from osgeo import gdal

# Function to resize numpy array
def resize_array(arr, size):
    return cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)

def normalize_array(arr):
    if arr.max() - arr.min() > 0:
        return (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def transform_and_paste_ship(composite_image, ship_image, ship_mask, terrain_mask, tx, ty, angle, label=False):
    h, w = ship_image.shape[:2]

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    # Apply translation
    M[0, 2] += tx
    M[1, 2] += ty

    # Apply affine transformation to ship image and mask
    rotated_ship = cv2.warpAffine(ship_image, M, (composite_image.shape[1], composite_image.shape[0]))
    rotated_mask = cv2.warpAffine(ship_mask.astype(np.uint8), M, (composite_image.shape[1], composite_image.shape[0]))

    # Combine masks
    final_mask = rotated_mask * terrain_mask

    # Apply the transformed ship onto the composite image
    composite_image[final_mask == 1] = rotated_ship[final_mask == 1]

    if label: # Add -1s where there is no data so its masked during training
        composite_image[terrain_mask == 0] = -1

    return composite_image

def generate_composites(terrain_folder, ship_folder, output_folder, total_ship_images, bag_xyz_dict, tag):
    '''
    Generates composite images using a masked ship in the foreground pasted on a terrain image.
    Each ship is used one time before repeating since there are only ~96. 
    Each terrain image is completely random.
    In progress: NOAA Ships are randomly place and rotated within the confines of the image. 
                 Field images are randomly placed and rotated within the confines of the masked region. 
    '''
    # Load file paths
    terrain_paths = [os.path.join(terrain_folder, f) for f in os.listdir(terrain_folder) if f.endswith('.npy')]
    ship_image_paths = [os.path.join(ship_folder, f) for f in os.listdir(ship_folder) if '_image' in f and f.endswith('.npy')]
    ship_label_paths = {os.path.join(ship_folder, f.replace('_image', '_label')): os.path.join(ship_folder, f.replace('_image', '_label')) for f in os.listdir(ship_folder) if '_image' in f and f.endswith('.npy')}

    

    # print(ship_image_paths)
    # print(ship_label_paths)

    # Ensure enough backgrounds
    if len(terrain_paths) == 0 or len(ship_image_paths) == 0:
        raise ValueError("Ensure both terrain and ship folders have valid .npy files.")

    # num400 = 0
    # num200 = 0
    # num100 = 0

    # Generate NOAA ship composite images
    count = 0
    while count < total_ship_images:
        random.shuffle(ship_image_paths)  # Ensure each ship is used before repeating

        for ship_image_path in ship_image_paths:
            if count >= total_ship_images:
                break

            # print("Ship image path", ship_image_path)
            if "IGLD" in ship_image_path or "LWD" in ship_image_path:
                ship_image_name = (str.split(ship_image_path, '/')[-1])[0:-10]
                check_name = "_".join((str.split(ship_image_name, '_'))[0:-1]) # Remove the actual name
            else:
                ship_image_name = (str.split(ship_image_path, '/')[-1])[0:-16]
                check_name = ship_image_name
            # print(ship_image_name)

            if check_name in bag_xyz_dict:
                res = int(200 / bag_xyz_dict[check_name])
                output_size = (res, res)
            else:
                output_size = (200, 200)

            # if output_size == (200, 200):
            #     num200 += 1
            # elif output_size == (400, 400):
            #     num400 += 1
            # else:
            #     num100 += 1

            ship_image = resize_array(np.load(ship_image_path), output_size)
            ship_label_path = ship_label_paths.get(ship_image_path.replace('_image', '_label'))
            # print("Ship label path", ship_label_path)
            if not ship_label_path or not os.path.exists(ship_label_path):
                continue

            ship_label = resize_array(np.load(ship_label_path), output_size)
            # ship_mask = (ship_label > 0).astype(np.uint8)
            ship_mask = ((ship_label == 1) & (ship_image != 0)).astype(np.uint8)
            ship_nonzero_mask = (ship_image != 0).astype(np.uint8)
            # print("ship label shape", ship_label.shape)

            terrain_path = random.choice(terrain_paths)
            terrain_image = resize_array(np.load(terrain_path), output_size)
            terrain_mask = (terrain_image != 0).astype(np.uint8)

            # Generate statistics for determining paste depth
            ship_pixels = ship_image[(ship_label == 1) & (ship_image != 0)]  # Ship pixels where label == 1
            terrain_pixels = terrain_image[terrain_mask == 1]
            # non_ship_pixels = ship_image[(ship_label == 0) & (ship_image != 0)]  # Nonzero background pixels

            # Compute means, avoiding division by zero
            mean_ship = np.mean(ship_pixels) if ship_pixels.size > 0 else 0
            mean_terrain = np.mean(terrain_pixels) if terrain_pixels.size > 0 else 0
            # depth_ratio = 0.9 # 0.87 is based on the average ship/terrain depth ratio from test set
            depth_ratio = np.random.normal(0.87, 0.06) # 0.91, 0.09
            desired_depth = depth_ratio * mean_terrain 
            shift_magnitude = desired_depth - mean_ship # -20 desired, -10 currently, -10 shift 
            # print(mean_ship, mean_terrain, shift_magnitude)

            composite_image = terrain_image.copy()
            h, w = ship_image.shape[:2]
            
            # if tag == 'NOAA': # place ship anywhere in image
            #     tx = random.randint(-w//2,w//2)
            #     ty = random.randint(-h//2,h//2)

            # elif tag == 'Field': # place ship only on valid pixels, find offset from center
            valid_indices = np.argwhere(terrain_mask)

            # Choose a random index from the valid indices
            if valid_indices.size == 0:
                print("No valid coordinates in the mask")
                continue
            
            random_index = np.random.randint(len(valid_indices)) # choose a random location within the data to place ship
            des_y, des_x = valid_indices[random_index]
            ty = des_y - h//2
            tx = des_x - w//2

            # else:
            #     print("Data tag not recognized")

            angle = random.randint(0,359) # end is included

            composite_label = np.zeros_like(ship_label, dtype = np.int32)
            transform_and_paste_ship(composite_image, ship_image + shift_magnitude, ship_mask, terrain_mask, tx, ty, angle)
            transform_and_paste_ship(composite_label, ship_label, ship_mask, terrain_mask, tx, ty, angle, label=True)

            composite_label[terrain_mask == 0] = -1

            # Save output
            img_name = f'GEN_{ship_image_name}_{tag}_{count:04d}_image.npy'
            label_name = f'GEN_{ship_image_name}_{tag}_{count:04d}_label.npy'

            np.save(os.path.join(output_folder, img_name), composite_image)
            np.save(os.path.join(output_folder, label_name), composite_label)

            count += 1

    print(f"Generated {count} ship images in {output_folder}.")
    # print(f"Generated {num400} 400px ships, {num200} 200px ships, and {num100} 100px ships")

def generate_terrain(terrain_folder, output_folder, total_terrain_images, output_size, tag):
    terrain_paths = [os.path.join(terrain_folder, f) for f in os.listdir(terrain_folder) if f.endswith('.npy')]

    # Save n terrain paths in addition to generated data to learn to ignore background terrain
    random.shuffle(terrain_paths)  # Ensure each terrain image is added before repeating (there are 1189)

    count = 0
    for terrain_path in terrain_paths[:total_terrain_images]:
        

        terrain_image = resize_array(np.load(terrain_path), output_size)
        np.save(os.path.join(output_folder, f'terrain_{tag}_{count:04d}_image.npy'), terrain_image)
        count += 1

    print(f"Generated {count} terrain images in {output_folder}.")

def get_bag_xyz_info(bag_xyz_directory):
    gdal.PushErrorHandler('CPLQuietErrorHandler') # Silence errors we don't care about
    output_dict = {}
    for file in os.listdir(bag_xyz_directory):
        if "IGLD" in file or "LWD" in file:
            name = (str.split(file, '/')[-1])[0:-4]
        else:
            name = (str.split(file, '/')[-1])[0:-10]
        
        filepath = os.path.join(bag_xyz_directory, file)
        try:
            gdalinfo = gdal.Info(filepath, format='json')
            # print(gdalinfo['geoTransform'][1])
            output_dict[name] = gdalinfo['geoTransform'][1]
        except Exception as e:
            # print(f"An error occurred on filename {name}")
            output_dict[name] = 1.0 # Everything with this error isn't in train set or has res 1.0
            continue
    gdal.PopErrorHandler()
    
    return output_dict

def main():

    random.seed(123456789) # Seed can be changed but needs to be consistent for all data generation

    # Paths to your data folders
    terrain_folder = 'NOAA_Terrain_Numpy_New'
    ship_folder = 'Train_Ships'
    output_folder = 'Generated_Data'
    field_terrain_folder = 'Field_Terrain'
    train_ship_bag_folder = 'Train_Ships_Bags'
    os.makedirs(output_folder, exist_ok=True)

    # Optional
    clear_directory(output_folder)

    # Parameters
    NOAA_ship_images = 400  # Total number of images to generate
    Field_ship_images = 200
    NOAA_terrain_images = 400
    Field_terrain_images = 200
    output_size = (200, 200)  # All terrain images are 1px resolution

    # Prep for different image resolutions
    bag_xyz_dict = get_bag_xyz_info(train_ship_bag_folder)

    # Generate field terrain, NOAA terrain
    generate_composites(terrain_folder, ship_folder, output_folder, NOAA_ship_images, bag_xyz_dict, tag = 'NOAA')
    generate_composites(field_terrain_folder, ship_folder, output_folder, Field_ship_images, bag_xyz_dict, tag = 'Field')
    generate_terrain(terrain_folder, output_folder, NOAA_terrain_images, output_size, tag = 'NOAA')
    generate_terrain(field_terrain_folder, output_folder, Field_terrain_images, output_size, tag = 'Field')

    # Add original training data
    copy_files('Train_Ships', output_folder)


if __name__ == "__main__":
    main()


