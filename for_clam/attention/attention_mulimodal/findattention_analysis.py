import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from clam_model import CLAM_SB  # Assuming clam_model is a module you have

# Initialize DataFrame to store image information
image_info_df = pd.DataFrame(columns=['filename', 'score', 'index', 'fold_number', 'serial_number'])

# Load results
#fold =6
for fold in range(6,7):

    results_path = f'/home/minkyoon/CLAM2/results2/update_multihead1_add_911_230915_forpaper_save_multihead/remission_multimodal_stratified_721_s1/split_{fold}_results.pkl'
    with open(results_path, 'rb') as file:
        results = pickle.load(file)

    # Initialize lists for true positives, true negatives, false positives, false negatives
 
    tp = []
    tn = []
    fp = []
    fn = []

    # Iterate over the dictionary items
    for slide_id, info in results.items():
        # Determine the predicted label (class with highest probability)
        pred_label = np.argmax(info['prob'])
        
        # Compare the predicted and true labels
        if pred_label == info['label']:
            # If they're the same, it's either a true positive or true negative
            if pred_label == 1:
                tp.append(int(slide_id))  # True positive
            else:
                tn.append(int(slide_id))  # True negative
        else:
            # If they're different, it's either a false positive or false negative
            if pred_label == 1:
                fp.append(int(slide_id))  # False positive
            else:
                fn.append(int(slide_id))  # False 
    

    for i in tp:
        model = CLAM_SB()
        model.set_classifier(49)
        state_dict_path = f'/home/minkyoon/CLAM2/results2/update_multihead1_add_911_230915_forpaper_save_multihead/remission_multimodal_stratified_721_s1/s_{fold}_checkpoint.pt'
        model.load_state_dict(torch.load(state_dict_path))
        
        # Load data
        data = torch.load(f'/home/minkyoon/crohn/csv/clam/remission/csvandpt/{i}.pt')
        tabular = pd.read_csv('/home/minkyoon/CLAM2/data/processed/remission_under_10.csv')
        tab = tabular.iloc[:, 1:][tabular['slide_id'] == i].iloc[:, :49].fillna(0)
        tensor = torch.FloatTensor(tab.values)
        
        # Model evaluation
        model.eval()
        # Move model and data to the same device
        device = torch.device("cpu")
        model = model.to(device)
        data = data.to(device)
        tensor = tensor.to(device)
            
        
        with torch.no_grad():
            _, _, _, _, _, A, _ = model(data, tensor)
        
        # Get top and bottom 4 attention scores and indices
        top_scores, top_indices = torch.topk(A, k=4, largest=True, dim=1)
        bottom_scores, bottom_indices = torch.topk(A, k=4, largest=False, dim=1)
        
        # Save images with highest and lowest attention scores
        data_csv = pd.read_csv(f'/home/minkyoon/crohn/csv/clam/remission/csvandpt/{i}.csv')
        for scores, indices, folder in [(top_scores, top_indices, 'tn_up'), (bottom_scores, bottom_indices, 'tn_down')]:
            indices = indices.cpu().numpy().astype(int)  # Move to CPU and convert to integer
            for score, index in zip(scores[0], indices[0]):
                npy_path = data_csv.iloc[index]['filepath']
                image = np.load(npy_path)
                score_str = f"{score:.4f}"
                filename = f'serial{i}_score{score_str}_index{index}_fold{fold}.png'
                title = f"Serial: {i}, Score: {score_str}"
                output_dir = f'/home/minkyoon/crohn/for_clam/attention/attention_mulimodal/image/{folder}'
                os.makedirs(output_dir, exist_ok=True)
                plt.imshow(image, cmap='gray')
                plt.title(title) 
                plt.axis('off')
                plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight', pad_inches=0)
                plt.close()
                
                # Update DataFrame
                new_row = pd.DataFrame({'filename': [filename], 'score': [score], 'index': [index], 'fold_number': [fold], 'serial_number': [i]})
                image_info_df = pd.concat([image_info_df, new_row]).reset_index(drop=True)

    # Save DataFrame to CSV
    image_info_df.to_csv(f'./results/image_info_fold{fold}_tn.csv', index=False)
