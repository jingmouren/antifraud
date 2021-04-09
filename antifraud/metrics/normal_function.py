from sklearn.metrics import confusion_matrix, roc_auc_score
import json
import numpy as np

def general_result(y_true, y_score, threshold=0.6):
    def pred(score, best_thresh):
        label = 0
        if score > best_thresh:
            label = 1
        return label
    y_score = np.array(y_score)
    if len(y_score.shape) == 2:
        y_score = y_score[:,1]
    # best_thresh = select_threshold(y_true, y_score)
    best_thresh = threshold
    y_pred = [pred(score, best_thresh) for score in y_score]
    c_m = confusion_matrix(y_true, y_pred)
    print("model works on the data, the confusion_matrix is:(Threshold:{})".format(str(best_thresh)), c_m)
    acc = (c_m[0, 0]+c_m[1, 1])/(c_m[0, 0]+c_m[0, 1]+c_m[1, 0]+c_m[1, 1])
    print("model works on the data, the accuracy is:", acc)
    pre = c_m[1, 1]/(c_m[1, 1]+c_m[0, 1])
    print("model works on the data, the precision is:", pre)
    re = c_m[1, 1]/(c_m[1, 1]+c_m[1, 0])
    print("model works on the data, the recall is:", re)
    f_score = (2*pre*re)/(pre+re)
    print("model works on the data, the F1-score is:", f_score)
    #train_label_binary = to_categorical(train_label)
    auc = roc_auc_score(y_true, y_score)
    print("model works on the data, the auc is:", auc)


def select_threshold(y_true, y_score):
    def pred(score, threshold):
        label = 0
        if score > threshold:
            label = 1
        return label
    best_th = 0
    f1_score = 0
    output = {'Precision':[], 'Recall':[]}
    for i in range(1,100):
        threshold = i/100
        y_pred = [pred(score, threshold) for score in y_score]
        c_m = confusion_matrix(y_true, y_pred)
        try:
            pre = c_m[1, 1]/(c_m[1, 1]+c_m[0, 1])
            re = c_m[1, 1]/(c_m[1, 1]+c_m[1, 0])
            output['Precision'].append(pre)
            output['Recall'].append((re))
            f_score = (2*pre*re)/(pre+re)
            if f_score>f1_score :
                f1_score = f_score
                best_th = threshold
        except:
            continue
    if len(output['Precision']) != 99:
        print("Unknown Error occurred when generate results.")
    with open('Precision_Recall.txt','w') as w:
        w.write(json.dumps(output))
    return best_th

