import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

lst_imgs = glob(r"E:\NVIDIA_AIC\Track4\Data\train\*.jpg")
df = pd.DataFrame(lst_imgs, columns=['image_path'])
df_train, df_val = train_test_split(df, test_size = 0.2, random_state=42)
df_train, df_test = train_test_split(df_train, test_size = 0.1, random_state = 42)

df_train['class_id'] = df_train.apply(lambda x: int(x.image_path.split("\\")[-1].split("_")[0]), axis = 1)
df_val['class_id'] = df_val.apply(lambda x: int(x.image_path.split("\\")[-1].split("_")[0]), axis = 1)
df_test['class_id'] = df_test.apply(lambda x: int(x.image_path.split("\\")[-1].split("_")[0]), axis = 1)

df_train.to_csv(r"data\train.csv", index=False)
df_val.to_csv(r"data\val.csv", index=False)
df_test.to_csv(r"data\test.csv", index=False)

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)