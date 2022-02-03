from hyper_param import *
from PIL import Image, ImageDraw, ImageFont
import time
import copy
from typing import List
from torchvision import transforms

def generate_colors(num_colors: int) -> np.array:
    """Generates an array with RGB triplets representing colors.

    Args:
        num_colors: the number of colors to generate.

    Returns:
        the generated colors.
    """

    np.random.seed(0)
    colors = np.random.uniform(0, 255, size=(num_colors, 3))
    time_in_ms = 1000 * time.time()
    np.random.seed(int(time_in_ms) % 2 ** 32)

    return colors

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
    font = ImageFont.truetype('C:\Windows\Font/arial.ttf', 25)
    image_with_bb = copy.deepcopy(image)
    painter = ImageDraw.Draw(image_with_bb)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = 'red'
        x_min, y_min, x_max, y_max = obtain_initial_coordinates(box)

        if normalized_coordinates:
            width, height = image.size
            x_min *= width
            y_min *= height
            x_max *= width
            y_max *= height

        coord_bb = [x_min, y_min, x_max, y_max]
        painter.rectangle(coord_bb, outline=color, width=4)
        painter.text((x_min, y_min - 30), text="Starfish", fill=color, font=font)

    return image_with_bb

def show_img(data,index_sample=9):

    classes_mi = data.classes
    num_mi_classes = len(classes_mi)
    colors_mi = generate_colors(num_mi_classes)

    image, target = data[index_sample]
    print(data.images[index_sample])
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

