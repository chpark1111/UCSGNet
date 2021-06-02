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
            translation_loss += torch.linalg.norm(eval.shift, dim=-1).square().clamp(min=0.5).sum(dim=1).mean(dim=0) #[batch_sz, num_shape, num_dim]

        out['param'] = parameter_loss
        out['trans'] = translation_loss

    temp_loss = 0.0
    if converter.alpha <= 0.05:
        for layer in csg_layer:
            temp_loss += (torch.abs(layer.temp).clamp_min(FLOAT_EPS) - FLOAT_EPS)
        out['temp'] = temp_loss

    convertor_loss = (torch.abs(converter.alpha).clamp_min(FLOAT_EPS) - FLOAT_EPS)
    out['conv'] = convertor_loss

    total_loss = (recon_loss + parameter_loss + lambda_t*translation_loss 
                            + lambda_alpha*convertor_loss + temp_loss*lambda_tau)
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



































































        