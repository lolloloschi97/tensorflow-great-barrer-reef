from hyper_param import *
from PIL import Image, ImageDraw, ImageFont
import time
import copy
from typing import List
from torchvision import transforms
from model_definition_retinanet import detect_objects

def obtain_initial_coordinates(bbox):
    xmin = int((bbox[0] - (bbox[2] / 2)) * IMAGE_WIDTH)
    ymin = int((bbox[1] - (bbox[3] / 2)) * IMAGE_HEIGHT)
    xmax = int((bbox[0] + (bbox[2] / 2)) * IMAGE_WIDTH)
    ymax = int((bbox[1] + (bbox[3] / 2)) * IMAGE_HEIGHT)
    return xmin, ymin, xmax, ymax

def draw_boxes(image: Image,
               boxes: List[List[float]],
               classes: List[str],
               labels: List[int],
               scores: List[float],
               colors: List[List[float]],
               normalized_coordinates: bool,
               add_text: bool = True) -> Image:
    """Draws a rectangle around each object together with the name of the category and the prediction score using a
    different color for each category.

    Args:
        image: the input image.
        boxes: the bounding boxes in the format [x_min, y_min, x_max, y_max]
               for all the objects in the image.
        classes: the name of the classes for all the objects in the image.
        labels: the labels for all the objects in the image.
        scores: the predicted scores for all the objects in the image..
        colors: the colors to use for each class of object.
        normalized_coordinates: if true the coordinates are multiplied
                                according to the height and width of the image.
        add_text: if true add a box with the name of the category and
                  the score.

    Returns:
        The generated image.
    """
    if WITH_COLAB:
        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf', 8)
    else:
        font = ImageFont.truetype('C:\Windows\Font/arial.ttf', 8)

    image_with_bb = copy.deepcopy(image)
    painter = ImageDraw.Draw(image_with_bb)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = 'red'
        x_min, y_min, x_max, y_max = box
        # x_min, y_min, x_max, y_max = obtain_initial_coordinates(box) ONLY IF RESHAPE

        coord_bb = [int(x_min), int(y_min), int(x_max), int(y_max)]
        painter.rectangle(coord_bb, outline=color, width=2)

        painter.text((x_min, y_min - 7), text="Starfish", fill=color, font=font)

    return image_with_bb

def show_img(data,index_sample=9):
    classes_mi = data.classes
    num_mi_classes = len(classes_mi)
    colors_mi = 'red'

    image, target = data[index_sample]
    boxes = target['boxes']
    labels = target['labels']
    classes = [data.classes[l.item()] for l in labels]
    image = transforms.ToPILImage()(image)
    cell_with_bb = draw_boxes(image,
                              boxes=boxes,
                              classes=classes,
                              labels=labels,
                              scores=[1.0] * len(boxes),
                              colors=colors_mi,
                              normalized_coordinates=False)

    fig, ax = plt.subplots()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cell_with_bb, aspect='auto')
    plt.show()


def show_prediction(image,model,data_mi_train,val=False,targets=[]):

    bounding_boxes, scores, categories, labels = detect_objects(image,
                                                                model,
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

    if val:
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
        fig.subplots_adjust(0, 0, 1, 0.9)
        ax.imshow(plot_image)
        ax.set_title("Prediction vs Groundtruth")
    else:
        plot_image = image_with_bb_pred
        fig, ax = plt.subplots()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

    ax.imshow(plot_image, aspect='auto')
    plt.show()