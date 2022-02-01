from data_loader import *
from hyper_param import *
from visualization import *
from dataset_class_generation import *


LOAD_PICKELS = True
TRAINING = True

def save_datasets(train_df, val_df):
    print("Renaming images..")
    rename_images()
    print("Images renamed!")
    print("Saving datasets...")
    train_df.to_pickle(TRAIN_ROOT + DATAFRAME_ROOT + "training_dataframe.pkl")
    val_df.to_pickle(VALIDATION_ROOT + DATAFRAME_ROOT + "validation_dataframe.pkl")
    train_df.to_csv(TRAIN_ROOT + DATAFRAME_ROOT + 'training.csv')
    val_df.to_csv(VALIDATION_ROOT + DATAFRAME_ROOT + 'validation.csv')
    print("Datasets saved")
    print("Moving images...")
    try:
        images_folder(train_df,TRAIN_ROOT + IMAGES_ROOT)
    except: print("train images has already been moved!")
    try:
        images_folder(val_df, VALIDATION_ROOT + IMAGES_ROOT)
    except: print("validation images has already been moved!")
    print("Images moved")


def load_datasets():
    print("Loading datasets...")
    train_set = pd.read_pickle(TRAIN_ROOT + DATAFRAME_ROOT + "training_dataframe.pkl")
    val_set = pd.read_pickle(VALIDATION_ROOT + DATAFRAME_ROOT + "validation_dataframe.pkl")
    print("Datasets loaded")
    return train_set, val_set

def main():
    if LOAD_PICKELS:
        training_df, validation_df = load_datasets()

    else:
        training_df, validation_df = data_loader(TRAIN_SIZE)
        save_datasets(training_df, validation_df)
        create_annotations(TRAIN_ROOT, "training.csv")
        create_annotations(VALIDATION_ROOT, "validation.csv")
        print('Creating label files...')
        print('Labels created')
        print("Dataset ready!")
    data_mi_transforms = {'train': transforms.Compose([transforms.ToTensor()]),
                          'val': transforms.Compose([transforms.ToTensor()])}


    data_mi_train = GreatBarrerReef_Dataset(path_folder= TRAIN_ROOT,
                                            ext_images="jpg",
                                            ext_annotations="txt",
                                            transforms=data_mi_transforms['train'])

    data_mi_val = GreatBarrerReef_Dataset(path_folder=VALIDATION_ROOT,
                                          ext_images="jpg",
                                          ext_annotations="txt",
                                          transforms=data_mi_transforms['val'])

    classes_mi = data_mi_train.classes
    num_mi_classes = len(classes_mi)
    index_sample = 9
    colors_mi = generate_colors(num_mi_classes)

    image, target = data_mi_train[index_sample]
    print(data_mi_train.images[index_sample])
    boxes = target['boxes']
    labels = target['labels']
    classes = [data_mi_train.classes[l.item()] for l in labels]
    image = transforms.ToPILImage()(image)
    cell_with_bb = draw_boxes(image,
                              boxes=boxes,
                              classes=classes,
                              labels=labels,
                              scores=[1.0] * len(boxes),
                              colors=colors_mi,
                              normalized_coordinates=False)

    plt.imshow(cell_with_bb)
    plt.tight_layout(pad=0)
    plt.margins(x=0)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()