from hyper_param import *
from tqdm import tqdm
from PIL import Image
from torchvision.ops import box_iou

def compute_voc_ap(recall, precision, use_07_metric=True):
    if use_07_metric:
        # use voc 2007 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                # get max precision  for recall >= t
                p = np.max(precision[recall >= t])
            # average 11 recall point precision
            ap = ap + p / 11.
    else:
        # use voc>=2010 metric,average all different recall precision as ap
        # recall add first value 0. and last value 1.
        mrecall = np.concatenate(([0.], recall, [1.]))
        # precision add first value 0. and last value 0.
        mprecision = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mprecision.size - 1, 0, -1):
            mprecision[i - 1] = np.maximum(mprecision[i - 1], mprecision[i])

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrecall[1:] != mrecall[:-1])[0]

        # sum (\Delta recall) * prec
        ap = np.sum((mrecall[i + 1] - mrecall[i]) * mprecision[i + 1])

    return ap


def compute_ious(a, b):
    """
    :param a: [N,(x1,y1,x2,y2)]
    :param b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """

    a = np.expand_dims(a, axis=1)  # [N,1,4]
    b = np.expand_dims(b, axis=0)  # [1,M,4]

    overlap = np.maximum(0.0,
                         np.minimum(a[..., 2:], b[..., 2:]) -
                         np.maximum(a[..., :2], b[..., :2]))  # [N,M,(w,h)]

    overlap = np.prod(overlap, axis=-1)  # [N,M]

    area_a = np.prod(a[..., 2:] - a[..., :2], axis=-1)
    area_b = np.prod(b[..., 2:] - b[..., :2], axis=-1)

    iou = overlap / (area_a + area_b - overlap)

    return iou


def calculate_score(
    preds: List[torch.Tensor],
    gts: List[torch.Tensor],
    iou_th: float
) -> float:
    num_tp = 0
    num_fp = 0
    num_fn = 0
    for p, GT in zip(preds, gts):
        if len(p) and len(GT):
            gt = GT.clone()
            gt[:, 2] = gt[:, 0] + gt[:, 2]
            gt[:, 3] = gt[:, 1] + gt[:, 3]
            pp = p.clone()
            pp[:, 2] = pp[:, 0] + pp[:, 2]
            pp[:, 3] = pp[:, 1] + pp[:, 3]
            iou_matrix = box_iou(pp, gt)
            tp = len(torch.where(iou_matrix.max(0)[0] >= iou_th)[0])
            fp = len(p) - tp
            fn = len(torch.where(iou_matrix.max(0)[0] < iou_th)[0])
            num_tp += tp
            num_fp += fp
            num_fn += fn
        elif len(p) == 0 and len(GT):
            num_fn += len(GT)
        elif len(p) and len(GT) == 0:
            num_fp += len(p)
    score = 5 * num_tp / (5 * num_tp + 4 * num_fn + num_fp)
    return score


def evaluate_voc(val_dataset, model, decoder, num_classes=20, iou_thread=0.5):
    preds, gts = [], []
    with torch.no_grad():
        with tqdm(val_dataset, unit="images") as tepoch:
            for idx_batch, (images, targets) in enumerate(tepoch):
                tepoch.set_description(f"Validation")
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                img = images[0]
                img.to(DEVICE)
                gt_bboxes, gt_classes = targets[0]['boxes'], targets[0]['labels']
                gt_bboxes, gt_classes = gt_bboxes.cpu(), gt_classes.cpu()
                gts.append([gt_bboxes, gt_classes])

                list_dict = model(images, targets)
                preds_scores , preds_classes , preds_boxes = list_dict[0]['scores'], list_dict[0]['labels'], list_dict[0]['boxes']
                preds_scores, preds_classes, preds_boxes = preds_scores.cpu(), preds_classes.cpu(), preds_boxes.cpu()

                preds_scores = preds_scores[preds_classes < 1]
                preds_boxes = preds_boxes[preds_classes < 1]
                preds_classes = preds_classes[preds_classes < 1]

                preds.append([preds_boxes, preds_classes, preds_scores])

            print("all val sample decode done.")

        all_ap = {}
        for class_index in tqdm(range(num_classes)):
            per_class_gt_boxes = [
                image[0][image[1] == class_index] for image in gts
            ]
            per_class_pred_boxes = [
                image[0][image[1] == class_index] for image in preds
            ]
            per_class_pred_scores = [
                image[2][image[1] == class_index] for image in preds
            ]

            fp = np.zeros((0, ))
            tp = np.zeros((0, ))
            scores = np.zeros((0, ))
            total_gts = 0

            # loop for each sample
            for per_image_gt_boxes, per_image_pred_boxes, per_image_pred_scores in zip(
                    per_class_gt_boxes, per_class_pred_boxes,
                    per_class_pred_scores):
                total_gts = total_gts + len(per_image_gt_boxes)
                # one gt can only be assigned to one predicted bbox
                assigned_gt = []
                # loop for each predicted bbox
                for index in range(len(per_image_pred_boxes)):
                    scores = np.append(scores, per_image_pred_scores[index])
                    if per_image_gt_boxes.shape[0] == 0:
                        # if no gts found for the predicted bbox, assign the bbox to fp
                        fp = np.append(fp, 1)
                        tp = np.append(tp, 0)
                        continue
                    pred_box = np.expand_dims(per_image_pred_boxes[index], axis=0)
                    iou = compute_ious(per_image_gt_boxes, pred_box)
                    gt_for_box = np.argmax(iou, axis=0)
                    max_overlap = iou[gt_for_box, 0]
                    if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                        fp = np.append(fp, 0)
                        tp = np.append(tp, 1)
                        assigned_gt.append(gt_for_box)
                    else:
                        fp = np.append(fp, 1)
                        tp = np.append(tp, 0)

            iou_ths = np.arange(0.3, 0.85, 0.05)
            f2_scores = [calculate_score(per_class_pred_boxes, per_class_gt_boxes, iou_th) for iou_th in iou_ths]
            # sort by score
            indices = np.argsort(-scores)
            fp = fp[indices]
            tp = tp[indices]
            # compute cumulative false positives and true positives
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            # compute recall and precision
            recall = tp / total_gts
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = compute_voc_ap(recall, precision)
            all_ap[class_index] = ap

        mAP = 0.
        for _, class_mAP in all_ap.items():
            mAP += float(class_mAP)
        mAP /= num_classes

    print('f2 score is: ', np.mean(f2_scores))

    return all_ap, mAP