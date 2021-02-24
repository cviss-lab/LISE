import pandas as pd

crop_df_pth = '../output/testing_minimum550_m_patch_1_ppb_900_crop_length_850_imagenet_noaug_20210212_032150/test.csv'
img_df_pth = '../datasets/PED_V2/2_processed/img_dataset.csv'

df_crop = pd.read_csv(crop_df_pth)
df_img = pd.read_csv(img_df_pth)

df_crop = df_crop[['original_fp']]
df_crop.drop_duplicates(inplace=True)

df = pd.merge(df_crop, df_img, on='original_fp', how='inner')
df.to_csv("output_img_df.csv", index=False)
