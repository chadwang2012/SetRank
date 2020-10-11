# This is the evaluation function
import numpy as np


def evaluate(postrain, posprobe, r, k):
    userlist = list(posprobe.keys())
    for user in userlist:
        r[user, postrain[user]] = -9999     # delete the items in training set

    pred = np.argsort(r, axis=1)[:, ::-1]

    recall = []
    precision = []
    map = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            ll = 1
            for l in range(kk):
                if pred[user, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    ll += 1
            recall_tmp.append(np.float(np.sum(predict_tmp > 0)) / len(posprobe[user]))
            precision_tmp.append(np.float(np.sum(predict_tmp > 0)) / kk)
            map_tmp.append(np.sum(predict_tmp / (np.array(range(kk)) + 1)) / kk)

        recall.append(np.mean(recall_tmp))
        precision.append(np.mean(precision_tmp))
        map.append(np.mean(map_tmp))

    return recall, precision, map


