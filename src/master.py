from data_loader import *
from hyper_param import *
from visualization import *
from evaluation import launch_tensorboard
from model_definition_retinanet import retina_net, execute, validate, detect_objects
from dataset_class_generation import *


LOAD_PICKELS = True
TRAINING = False

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

    # Dataset class initialization

    #resized_height = int(IMAGE_HEIGHT/RESHAPE_FACTOR)
    #resized_width = int(IMAGE_WIDTH/RESHAPE_FACTOR)
    #transforms.Resize((resized_height, resized_width))
    data_mi_transforms = {'train': transforms.Compose([transforms.ToTensor()]),
                          'val': transforms.Compose([transforms.ToTensor()])}


    data_mi_train = GreatBarrerReef_Dataset(path_folder= TRAIN_ROOT,
                                            ext_images="jpg",
                                            ext_annotations="txt",
                                            transforms=data_mi_transforms['train'])

    data_mi_val = GreatBarrerReef_Dataset(path_folder=VALIDATION_ROOT,
                                          ext_images="jpg",
                                          ext_annotations="txt",
                                          transforms=data_mi_transforms['val'],
                                          train=False)

    # visualize image
    show_img(data_mi_train)


    # Data loaders
    loader_mi_train = torch.utils.data.DataLoader(data_mi_train,
                                                  batch_size= BATCH_SIZE,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers= NUM_WORKER,
                                                  collate_fn= collate_fn)

    loader_mi_val = torch.utils.data.DataLoader(data_mi_val,
                                                batch_size= 1,
                                                shuffle=False,
                                                num_workers= NUM_WORKER,
                                                collate_fn= collate_fn)

    name_train = "retina_net"
    launch_tensorboard(name_train)

    if TRAINING:
        execute(name_train, retina_net, LR, EPOCHS, loader_mi_train, loader_mi_val)
    else:
        retina_net.load_state_dict(torch.load(CHECKPOINT_ROOT + '/retina_net_10_epochs.bin'))
        all_ap_epoch, mAP_epoch = validate(retina_net, loader_mi_val, DEVICE)


    image, targets = data_mi_val[100]
    batch_image = image.unsqueeze(0)
    image = image.unsqueeze(0).to(DEVICE)

    bounding_boxes, scores, categories, labels = detect_objects(image,
                                                                retina_net,
                                                                0.25,
                                                                data_mi_train.classes)
    image = transforms.ToPILImage()(image[0])
    image_with_bb_pred = draw_boxes(image,
                                    bounding_boxes,
                                    categories,
                                    labels,
                                    scores,
                                    'red',
                                    normalized_coordinates=False,
                                    add_text=False)

    image_with_bb_gt = draw_boxes(image,
                                  targets["boxes"],
                                  categories,
                                  targets["labels"],
                                  [1.0] * len(targets["boxes"]),
                                  'red',
                                  normalized_coordinates=False,
                                  add_text=False)

    plot_image = np.concatenate((image_with_bb_pred, image_with_bb_gt), axis=1)
    fig, ax = plt.subplots(figsize=plt.figaspect(plot_image))
    plt.axis('off')
    fig.subplots_adjust(0, 0,1, 0.9)
    ax.imshow(plot_image)
    ax.set_title("Prediction vs Groundtruth")
    plt.show()


if __name__ == '__main__':
    main()