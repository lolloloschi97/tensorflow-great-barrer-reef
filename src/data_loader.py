import shutil
import os
from sklearn.model_selection import train_test_split
from hyper_param import *

def data_loader(TRAIN_SIZE):
    path = DATASET_ROOT + DATAFRAME_ROOT + "train.csv"
    df = pd.read_csv(path)
    if WITH_COLAB:
        df = df.loc[df['video_id']==0]
    indexes = train_test_split(np.arange(df.shape[0]),train_size = TRAIN_SIZE, random_state = 1)
    training_df = df.loc[indexes[0],:]
    validation_df = df.loc[indexes[1],:]
    return training_df, validation_df

def rename_images():
    for el in range(len(os.listdir(DATASET_ROOT + IMAGES_ROOT))-1):
        path = DATASET_ROOT + IMAGES_ROOT + "video_" + str(el) + "/"
        for image in os.listdir(path):
            old_file_name = path + image
            new_file_name = path + str(el) + "-" + image.split(".")[0] + ".jpg"
            os.rename(old_file_name, new_file_name)

def images_folder(df, destination_path):
    path = DATASET_ROOT + IMAGES_ROOT + "video_"
    df_files = [path + str(image_id.split("-")[0]) + "/" + str(image_id) + ".jpg"
                    for image_id in df.loc[:,"image_id"]]
    for f in df_files:
        shutil.move(f, destination_path)



