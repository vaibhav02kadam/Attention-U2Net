import os
import numpy as np
import cv2

def load_images(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        img = img/255
        if img is not None:
            images.append(img)
    return images


def calculate_precision(predicted_outputs, ground_truths):

    predicted_outputs = predicted_outputs > 0.5
    ground_truths = ground_truths > 0.5

    # assert predicted_outputs.shape == ground_truths.shape, "Both masks should have the same shape."

    # Calculate true positives (TP)
    TP = np.sum(np.logical_and(predicted_outputs == 1, ground_truths == 1))

    # Calculate false positives (FP)
    FP = np.sum(np.logical_and(predicted_outputs == 1, ground_truths == 0))

    # Calculate precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return precision

def calculate_recall(predicted_outputs, ground_truths):

    predicted_outputs = predicted_outputs > 0.5
    ground_truths = ground_truths > 0.5

    # assert predicted_outputs.shape == ground_truths.shape, "Both masks should have the same shape."

    # Calculate true positives (TP)
    TP = np.sum(np.logical_and(predicted_outputs == 1, ground_truths == 1))

    # Calculate false negatives (FN)
    FN = np.sum(np.logical_and(predicted_outputs == 0, ground_truths == 1))
 
    # Calculate recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return recall

def calculate_mae(predicted_outputs, ground_truths):
    mae_values = []
    for pred, gt in zip(predicted_outputs, ground_truths):
        mae = np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))
        # print("mae: = ", mae)
        mae_values.append(mae)
    return mae_values

def calculate_fbeta_score(predicted_outputs, ground_truths, beta):
    fbeta_values = []
    for pred, gt in zip(predicted_outputs, ground_truths):
        precision = calculate_precision(pred, gt)
        recall = calculate_recall(pred, gt)
        # print("Recall: = ", recall)
        # print("Precision: =", precision)
        if precision + recall == 0:
            fbeta_score = 0
        else:
            fbeta_score = (1 + beta) * (precision * recall) / (beta * precision + recall)
        fbeta_values.append(fbeta_score)

    return np.mean(fbeta_values)

predicted_outputs_folder = "/home/pvnatu/mihir/dl/Attention_U2Net/data/DUTS_test/u2net_results"
ground_truths_folder = "/home/pvnatu/mihir/dl/Attention_U2Net/data/DUTS_test/DUTS_testlabel"

predicted_outputs = load_images(predicted_outputs_folder)
ground_truths = load_images(ground_truths_folder)

mae_values = calculate_mae(predicted_outputs, ground_truths)
fbera_score = calculate_fbeta_score(predicted_outputs, ground_truths, 0.3)

average_mae = np.mean(mae_values)
print(f"Average MAE for all 1000 images: {average_mae:.4f}")
print("Fbeta: = ", fbera_score)