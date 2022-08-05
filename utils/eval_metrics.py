from torchmetrics import IoU
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
import numpy as np
import torch

# import train

# Utils
LABELS=['No Buildings','Buildings']

def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(gts,predictions,labels=range(len(label_values)))
    print("Confusion matrix :")
    print(cm)
    print("------------------------------------")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("------------------------------------")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("----------------------------")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    
    print("----------------------------")
    
    #compute jaccard distance/ for similarity it is 1-J
    """intersection is equivalent to True Positive count
    union is the mutually inclusive area of all labels & predictions"""
    smooth = 1e-7
    intersection = (predictions * gts).sum()
    total = (predictions + gts).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    print(f"Manually computed Jaccard Index:{1-IoU}")
    
    #nxt method
    j_index = jaccard_score(y_true=gts,y_pred=predictions, average='micro')
    print(f"Jaccard Index is:{j_index}")

    return accuracy,kappa,j_index

def f1_score(y_true:torch.Tensor, y_pred:torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    y_pred = torch.round(y_pred)
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    acc = (tp+tn)/(tp+fn+tn+fp)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    
    
    #calculate jaccard index
    intersection = (y_pred * y_true).sum()
    total = (y_pred + y_true).sum()
    union = total - intersection
    IoU = (intersection + epsilon) / (union + epsilon)     

    #calculating cohen's kappa
    Po = acc
    Pc = ((tp+fn)/(tp+fn+tn+fp))/((tp+fp)/(tp+fn+tn+fp))
    Pi = ((fp+tn)/(tp+fn+tn+fp))/((tn+fn)/(tp+fn+tn+fp))
    Pe = Pc+Pi
    K = (Po-Pe)/(1-Pe)
    return f1,acc,IoU,K
# if __name__ == "__main__":
#     metrics()