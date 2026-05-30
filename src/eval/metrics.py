from sklearn.metrics import f1_score, recall_score
HATE, NON_HATE = 1, 0

def evaluate(preds, labels):
    '''모든 실험은 오직 이 함수만 호출한다.'''
    return {"macro_f1":   f1_score(labels, preds, average="macro"),
            "hate_recall": recall_score(labels, preds, pos_label=HATE)}

def evaluate_by_group(preds, labels, groups):
    out = {}
    for g in set(groups):
        idx = [k for k, gi in enumerate(groups) if gi == g]
        l = [labels[k] for k in idx]; p = [preds[k] for k in idx]
        out[g] = {"hate_recall": recall_score(l, p, pos_label=HATE) if HATE in l else None,
                  "n": len(p)}
    return out
