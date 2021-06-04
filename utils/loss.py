import torch
import torch.nn
import torch.linalg
import torch.nn.functional as F
import numpy as np
import cv2

FLOAT_EPS = torch.finfo(torch.float32).eps

def total_loss(pred, gt, converter, csg_layer, evaluators, uses_planes):
    lambda_tau = lambda_t = lambda_alpha = 0.1
    total_loss = 0.0
    out = dict()

    recon_loss = F.mse_loss(pred, gt)
    out['rec'] = recon_loss

    parameter_loss = 0.0
    translation_loss = 0.0
    if not uses_planes:
        for eval in evaluators:
            parameter_loss += (-eval.param).clamp(min = 0.0).sum(dim=(1, 2)).mean(dim=0) #[batch_sz, num_shape, num_param]
            translation_loss += ((eval.shift_vector_prediction().norm(dim=-1) - 0.5).relu() ** 2).mean() #[batch_sz, num_shape, num_dim]

        translation_loss = lambda_t*translation_loss 
        out['param'] = parameter_loss
        out['trans'] = translation_loss

    temp_loss = 0.0
    if converter.alpha <= 0.05:
        for layer in csg_layer:
            temp_loss += (torch.abs(layer.temp).clamp_min(FLOAT_EPS) - FLOAT_EPS)
    temp_loss = lambda_tau*temp_loss
    out['temp'] = temp_loss

    convertor_loss = lambda_alpha*(torch.abs(converter.alpha).clamp_min(FLOAT_EPS) - FLOAT_EPS)
    out['conv'] = convertor_loss

    total_loss = (recon_loss + parameter_loss + translation_loss 
                            + convertor_loss + temp_loss)
    return total_loss, out

'''
Chamfer_distance by
https://github.com/Hippogriff/CSGNet
'''
def chamfer(images1, images2):
    """
    Chamfer distance on a minibatch, pairwise.
    :param images1: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :param images2: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :return: pairwise chamfer distance
    """
    # Convert in the opencv data format
    images1 = images1.astype(np.uint8)
    images1 = images1 * 255
    images2 = images2.astype(np.uint8)
    images2 = images2 * 255
    N = images1.shape[0]
    size = images1.shape[-1]

    D1 = np.zeros((N, size, size))
    E1 = np.zeros((N, size, size))

    D2 = np.zeros((N, size, size))
    E2 = np.zeros((N, size, size))
    summ1 = np.sum(images1, (1, 2))
    summ2 = np.sum(images2, (1, 2))

    # sum of completely filled image pixels
    filled_value = int(255 * size**2)
    defaulter_list = []
    for i in range(N):
        img1 = images1[i, :, :]
        img2 = images2[i, :, :]

        if (summ1[i] == 0) or (summ2[i] == 0) or (summ1[i] == filled_value) or (summ2[\
                i] == filled_value):
            # just to check whether any image is blank or completely filled
            defaulter_list.append(i)
            continue
        edges1 = cv2.Canny(img1, 1, 3)
        sum_edges = np.sum(edges1)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue
        dst1 = cv2.distanceTransform(
            ~edges1, distanceType=cv2.DIST_L2, maskSize=3)

        edges2 = cv2.Canny(img2, 1, 3)
        sum_edges = np.sum(edges2)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue

        dst2 = cv2.distanceTransform(
            ~edges2, distanceType=cv2.DIST_L2, maskSize=3)
        D1[i, :, :] = dst1
        D2[i, :, :] = dst2
        E1[i, :, :] = edges1
        E2[i, :, :] = edges2
    distances = np.sum(D1 * E2, (1, 2)) / (
        np.sum(E2, (1, 2)) + 1) + np.sum(D2 * E1, (1, 2)) / (np.sum(E1, (1, 2)) + 1)
    
    distances = distances / 2.0
    # This is a fixed penalty for wrong programs
    distances[defaulter_list] = 16
    return np.mean(distances, axis=0)

def iou(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    iou_val = (y_true * y_pred).sum(axis=(1, 2)) / ((y_true + y_pred).clip(0, 1).sum(axis=(1, 2)) + 1.0)
    return np.mean(iou_val, axis=0)

def chamfer_distance_3d(gt_points: torch.Tensor, pred_points: torch.Tensor):
    gt_num_points = gt_points.shape[0]
    pred_num_points = pred_points.shape[0]

    points_gt_matrix = gt_points.unsqueeze(1).expand(
        [gt_points.shape[0], pred_num_points, gt_points.shape[-1]]
    )
    points_pred_matrix = pred_points.unsqueeze(0).expand(
        [gt_num_points, pred_points.shape[0], pred_points.shape[-1]]
    )

    distances = (points_gt_matrix - points_pred_matrix).pow(2).sum(dim=-1)
    match_pred_gt = distances.argmin(dim=0)
    match_gt_pred = distances.argmin(dim=1)

    dist_pred_gt = (pred_points - gt_points[match_pred_gt]).pow(2).sum(dim=-1).mean()
    dist_gt_pred = (gt_points - pred_points[match_gt_pred]).pow(2).sum(dim=-1).mean()

    chamfer_distance = dist_pred_gt + dist_gt_pred

    return chamfer_distance.item()