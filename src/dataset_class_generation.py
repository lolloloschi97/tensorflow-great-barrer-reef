from hyper_param import *
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

def obtain_corners_yolo(bbox):
    x_center = (bbox['x'] + int(bbox['width']/2)) / IMAGE_WIDTH
    y_center = (bbox['y'] + int(bbox['height']/2)) / IMAGE_HEIGHT
    w = bbox['width'] / IMAGE_WIDTH
    h = bbox['height'] / IMAGE_HEIGHT
    return x_center, y_center, w, h

def obtain_corners(bbox):
    x_min = bbox['x']
    y_min = bbox['x']
    x_max = bbox['x'] + bbox['width']
    y_max = bbox['x'] + bbox['height']

    return x_min, y_min, x_max, y_max

def create_annotations(df_path,df_name):
    df_training = pd.read_csv(df_path + DATAFRAME_ROOT + df_name)
    for row in range(df_training.shape[0]):
        image_id = df_training.loc[row,"image_id"]
        annotations = df_training.loc[row,"annotations"]
        annotations = json.loads(annotations.replace("'",'"'))
        if len(annotations) == 0:
            file = open(df_path + "labels/" + image_id + ".txt", mode="w")
            file.write("__background__ 1 2 3 4")
        else:
            file = open(df_path + "labels/" + image_id + ".txt", mode="w")
            for ann in range(len(annotations)):
                xmin, ymin, xmax, ymax = obtain_corners(annotations[ann])
                file.write("Starfish " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")

                #x_center, y_center, w, h = obtain_corners_yolo(annotations[ann])
                #file.write("Starfish " + str(x_center) + " " + str(y_center) + " " + str(w) + " " + str(h) + "\n")

def parse_annotations_file(path_to_file: str):
    """Parse annotation file generated with the OIDv4 ToolKit.

    Args:
        path_to_file: the path to the file with the annotations.

    Returns:
        The classes for each object in the image.
        The bounding boxes coordinates for each object in the image in
        [x_min, y_min, x_max, y_max] format, with values between 0 and H and 0 and W.
    """
    with open(path_to_file) as file_annotations:
        obj_classes, boxes  = [], []
        for annotation in file_annotations:
            category, x_min, y_min, x_max, y_max = annotation.rstrip().split(" ")[-5:]
            obj_classes.append(category)

            try:
                coordinates = [float(x_min), float(y_min), float(x_max), float(y_max)]
            except ValueError:
                print(f'Error in converting float to string for the line {coordinates}')
                raise

            boxes.append(coordinates)
    return obj_classes, boxes

def collate_fn(batch):
    return tuple(zip(*batch))


class GreatBarrerReef_Dataset(Dataset):
    def __init__(self,
        path_folder: str,
        ext_images: str,
        ext_annotations: str,
        transforms: torchvision.transforms = None,
    ) -> None:
        """Init the dataset

        Args:
            path_images: the path to the folder containing the images.
            ext_images: the extension of the images.
            ext_annotations: the extension of the annotations.
            transforms: the transformation to apply to the dataset.
        """
        self.images = sorted([path for path in Path(path_folder + IMAGES_ROOT).rglob(f"*.{ext_images}")])
        self.annotations = sorted(
            [path for path in Path(path_folder + LABELS_ROOT).rglob(f"*.{ext_annotations}")]
        )
        self.transforms = transforms

        self.classes = ["__background__", "Starfish"]
        if len(self.images) - len(self.annotations) != 0:
            raise AssertionError(
                f"Labels and Images differs in size: {len(self.images)} - {len(self.annotations)}."
            )

    def __getitem__(self, idx):
        path_image = self.images[idx]
        path_annotations = self.annotations[idx]
        image = Image.open(path_image).convert("RGB")
        classes, boxes = parse_annotations_file(path_annotations)

        boxes = [[b[0], b[1], b[2], b[3]] for b in boxes]
        labels = [self.classes.index(c) for c in classes]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.images)



