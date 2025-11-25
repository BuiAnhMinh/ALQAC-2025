def inspect_question(idx, x_val_df, y_true, scores, threshold):
    """
    idx        : index of the question in validation set
    x_val_df   : validation DataFrame or Series containing texts
    y_true     : binary matrix Y_val
    scores     : val_scores (probability matrix)
    threshold  : chosen threshold

    Prints debugging info for one question.
    """

    print("="*80)
    print(f"QUESTION INDEX: {idx}")
    print("-"*80)
    print("QUESTION TEXT:")
    print(x_val_df.iloc[idx])
    print()

    # true labels
    true_indices = np.where(y_true[idx] == 1)[0]
    true_labels = [mlb.classes_[j] for j in true_indices]

    # predicted scores for all labels
    row = scores[idx]

    # predicted labels above threshold
    pred_indices = np.where(row >= threshold)[0]
    pred_labels = [mlb.classes_[j] for j in pred_indices]

    # fallback if none
    if len(pred_indices) == 0:
        best_idx = np.argmax(row)
        pred_indices = [best_idx]
        pred_labels = [mlb.classes_[best_idx]]
        used_fallback = True
    else:
        used_fallback = False

    print("TRUE LABELS:")
    print(true_labels)
    print()

    print("MODEL SCORES (top 10):")
    top10 = np.argsort(-row)[:10]
    for j in top10:
        print(f"{mlb.classes_[j]:40s}  score={row[j]:.4f}")
    print()

    print(f"THRESHOLD USED = {threshold}")
    print("PREDICTED LABELS:")
    print(pred_labels)
    print()

    if used_fallback:
        print("⚠️ Fallback used: model predicted nothing above threshold.")
        print()

    # compute TP, FP, FN
    true_set = set(true_indices)
    pred_set = set(pred_indices)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0
    beta = 2
    beta2 = beta**2
    if precision == 0 and recall == 0:
        f2 = 0
    else:
        f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall)

    print("EVALUATION:")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F2-score  = {f2:.4f}")
    print("="*80)
    
    