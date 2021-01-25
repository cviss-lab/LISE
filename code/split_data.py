from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import glob, os

pth="../datasets/PED/2_detected_imgs"

validation_split = 0.2
random_state=42
split_methods = ['normal', 'by_scene']
split_method = 'by_scene'
train_methods = ['normal', 'k_folds']
train_method = 'k_folds'

lable_pths = glob.glob(os.path.join(pth, '**/*.csv'), recursive=True)
for lable_pth in lable_pths:
    labels = pd.read_csv(lable_pth)
    # Get the number of
    split_labels = [[pth.split('/')[-3], pth.split('/')[-2]] for pth in labels['original_fp'].values]
    df = pd.DataFrame(split_labels, columns=['scene', 'num'])
    scenes = df['scene'].unique()
    scene_df = df.drop_duplicates()

    for scene in scenes:
        # Ordered keywords of dataset
        s = scene_df[scene_df['scene'] == scene]
        # Total number of scenes
        num_scenes = len(s)
        # Total number of test scenes
        num_test_scenes = int(validation_split*num_scenes)
        test_scene_ind = np.random.choice(s['num'].values, num_test_scenes, replace=False)
        test_scenes = s['num'][s['num'].isin(test_scene_ind)]
        train_scenes = s['num'][~s['num'].isin(test_scene_ind)]
        # Create dataframes
        test_df = labels[(df['scene'] == scene) & (df['num'].isin(test_scenes.values))]
        train_df = labels[((df['scene'] == scene) & (~df['num'].isin(test_scenes.values)))]
        # Export
        f_name = lable_pth.split('/')[-1]
        folder_pth = lable_pth.split('/')[:-1]
        if len(folder_pth) > 1:
            folder_pth = '/'.join(folder_pth)
        else:
            folder_pth = folder_pth[0]
        test_df.to_csv(f"{folder_pth}/test_{scene}_{f_name}", index=False)
        train_df.to_csv(f"{folder_pth}/train_{scene}_{f_name}", index=False)
