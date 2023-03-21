import torch


def general_postprocessing(prediction):
    try:
        if type(prediction).__name__ == 'ImagePatch':
            prediction = prediction.classify_object()

        if isinstance(prediction, list):
            prediction = prediction[0] if len(prediction) > 0 else "no"

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.item()
        if prediction is None:
            prediction = "no"
        if isinstance(prediction, bool):
            if prediction:
                prediction = "yes"
            else:
                prediction = "no"
        elif isinstance(prediction, int):
            prediction = str(prediction)
            print("No answer is a number, so this will be wrong")
    except:
        prediction = str(prediction)

    prediction = str(prediction)

    prediction = prediction.replace('\n', ' ')
    prediction = prediction.replace('\t', ' ')
    prediction = prediction.strip()
    prediction = prediction.lower()

    if prediction == 'true':
        prediction = 'yes'
    elif prediction == 'false':
        prediction = 'no'
    return prediction


def accuracy(prediction, ground_truth, *args):
    """
    Args:
        prediction (list): List of predicted answers.
        ground_truth (list): List of ground truth answers.
    Returns:
        score (float): Score of the prediction.
    """
    if len(prediction) == 0:  # if no prediction, return 0
        return 0
    assert len(prediction) == len(ground_truth)
    pred_gt_filtered = [(pred, gt) for pred, gt in zip(prediction, ground_truth) if gt != '']
    score = 0
    for p, g in pred_gt_filtered:
        if general_postprocessing(p) == g:
            score += 1
    return score / len(pred_gt_filtered)
