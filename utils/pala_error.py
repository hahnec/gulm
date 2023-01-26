import numpy as np

jaccard_index = lambda tp, fn, fp: tp/(fn+tp+fp)

def rmse_unique(pt_array, gt_array, tol=1/4):

    if pt_array.size == 0:
        return float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN'), float('NaN')

    mask = np.ones(len(pt_array), dtype=bool)
    fn_num = 0

    l2_errs = []
    for gt in gt_array:
        l2_distances = ((pt_array-gt)**2).sum(-1)**.5
        l2_err, idx = np.min(l2_distances), np.argmin(l2_distances)
        if l2_err < tol:# and mask[idx]:
            l2_errs.append(l2_err)
            mask[idx] = False
        else:
            fn_num += 1
    
    fp_num = sum(mask)
    tp_num = len(l2_errs)

    jaccard = jaccard_index(tp_num, fn_num, fp_num) * 100
    precision = tp_num/(fp_num+tp_num) * 100 if fp_num+tp_num != 0 else 0
    recall = tp_num/(fn_num+tp_num) * 100 if fn_num+tp_num != 0 else 0

    return np.mean(l2_errs), precision, recall, jaccard, tp_num, fp_num, fn_num
