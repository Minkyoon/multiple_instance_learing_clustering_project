import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path 

def get_directory_path(folder):
    return list(folder.glob("*"))

def get_directory_path_in_list(path_list):
    data_list = []
    for path in path_list:
        data_list += get_directory_path(path)
    return data_list

def get_data_path(folder):
    return list(folder.glob("*.dcm"))
# get data path that is in list path folder

def get_data_path_in_list(path_list):
    data_list = []
    for path in path_list:
        data_list += get_data_path(path)
    return data_list

def remove_black_area(image : np.array, tol : int = 20) :
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_image>tol
    return image[np.ix_(mask.any(1),mask.any(0))]

def circular_crop_and_mask(image: np.array):
    height, width, _ = image.shape
    mask = np.zeros((height, width), np.uint8)
    cx, cy = width // 3, height // 2  # Change the center x-coordinate
    radius = cy  # Change the radius to half of the smaller dimension
    ratio = min(1.18, image.shape[1] / image.shape[0])
    cy2 = int(cy *ratio)

    # Create circular mask
    cv2.circle(mask, (cx, cy), int(radius*0.8), 255, -1)  # Remove the 0.9 multiplier from the radius
    circular_masked_img = cv2.bitwise_and(image, image, mask=mask)

    # Crop the rectangular bounding box around the circle
    cropped_img = circular_masked_img[max(0, cy - radius):min(height, cy + radius), max(0, cx - radius):min(width, cx + radius)]


    # Crop the rectangular bounding box around the circle
    cropped_img = circular_masked_img[cy - radius:cy + radius, width -cy2 - radius:width -cy2 + radius]

    return cropped_img, circular_masked_img

def mornalize_img(arr: np.array):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def main(raw_data_path):
    # file load
    
    directory_list = get_directory_path_in_list(raw_data_path)
    
    for i in range(len(directory_list)):
        data_list = get_data_path(directory_list[i])
        for j in range(len(data_list)):
            try:
                dcm_path = data_list[j]
                dcm = pydicom.dcmread(dcm_path)
                accession_num = dcm.AccessionNumber
                inorder_num = dcm.InstanceNumber
                patient_id = dcm.PatientID
                img = dcm.pixel_array
                str_path = str(dcm_path)
                split_path = str_path.split('/')
                desired_part = split_path[-2]
                index_num = desired_part.split('_')[0]
                


                # color fix
                if np.all(img[0][0] == np.array([0, 128, 128])):
                    color_fixed_image = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
                else:
                    color_fixed_image = img
                # remove black area
                removed_colorfixed_image = remove_black_area(color_fixed_image)
                # crop and mask circular endoscopic area
                cropped_removed_colorfixed_image, circular_masked_image = circular_crop_and_mask(removed_colorfixed_image)
                
                
                resized_cropped_image = cv2.resize(cropped_removed_colorfixed_image, (225, 225), interpolation=cv2.INTER_AREA)
                norm_img = mornalize_img(resized_cropped_image)
                createDirectory(f'/home/minkyoon/2023_crohn_data/processed_data4/{index_num}')
                np.save(f'/home/minkyoon/2023_crohn_data/processed_data4/{index_num}/{inorder_num}.npy', norm_img)
                norm_img
            except Exception as e:
                print(f"An error occurred at index i={i}, j={j}: {e}")
                with open("error_log4.txt", "a") as file:
                    file.write(f"An error occurred at index i={i}, j={j} with file {dcm_path}: {e}\n")
                continue
                
                
   
if __name__ == "__main__":
    
    raw_data_path = [Path('/home/minkyoon/2023_crohn_data/original_data')]
    main(raw_data_path)
    







