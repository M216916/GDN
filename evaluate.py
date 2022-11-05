from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def get_full_err_scores(test_result, val_result):
    np_test_result = np.array(test_result)                         # (3, 2044, 27)
    np_val_result = np.array(val_result)                           # (3,  312, 27)

    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1]                         # 27
    labels = np_test_result[2, :, 0].tolist()                      # len : 2044 (0.0 or 1.0)

    for i in range(feature_num):
        test_re_list = np_test_result[:2,:,i]                      # (2, 2044)
        val_re_list = np_val_result[:2,:,i]                        # (2,  312)

        scores = get_err_scores(test_re_list, val_re_list)         # (2044,)
        normal_dist = get_err_scores(val_re_list, val_re_list)     # ( 312,)

        if all_scores is None:                                     # i = 0 の時実行
            all_scores = scores
            all_normals = normal_dist
        else:                                                      # i = 1 ~ 26 の時実行
            all_scores = np.vstack((all_scores, scores))           # all_scores  : (2, 2044) → (3, 2044) → ... → (27, 2044)
            all_normals = np.vstack((all_normals, normal_dist))    # all_normals : (2,  312) → (3,  312) → ... → (27,  312)

    return all_scores, all_normals



def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])
    
    return smoothed_err_scores


def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1):       # total_err_scores : (27, 2044) ／ gt_labels : len(2044) (0.0 or 1.0)

    total_features = total_err_scores.shape[0]                            # 27

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
                                                                          # (1, 2044)
    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)
                                                                          # (2044,)
    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)
                                                                          # final_topk_fmeas : len(400) ／ thresolds : len(400)
    th_i = final_topk_fmeas.index(max(final_topk_fmeas))                  # th_i : final_topk_fmeas の最大値のindex
    thresold = thresolds[th_i]                                            # thresoldsの index 番目

    pred_labels = np.zeros(len(total_topk_err_scores))                    # (0, 0, 0, ... ,0) : len(2044)
    pred_labels[total_topk_err_scores > thresold] = 1                     # 条件を満たす部分を 1 に

    for i in range(len(pred_labels)):                                     # pred_labels ／ gt_labels を整数に
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)                         # precision(適合率)
    rec = recall_score(gt_labels, pred_labels)                            # recall(再現率)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)           # auc

    return max(final_topk_fmeas), pre, rec, auc_score, thresold
