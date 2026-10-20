"""Nested leave-one-run-out training loop shared by GCN and GAT experiments."""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from .graphs import make_graph_dataset


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        ys.append(batch.y.cpu().numpy())
        ps.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred), y_true, y_pred


def class_weights_tensor(y_tr, n_classes, device):
    cls = np.unique(y_tr)
    weights = compute_class_weight(class_weight="balanced", classes=cls, y=y_tr)
    full_w = np.ones(n_classes, dtype=np.float32)
    for c, w in zip(cls, weights):
        full_w[int(c)] = w
    full_w = np.sqrt(full_w)
    full_w = full_w / full_w.mean()
    return torch.tensor(full_w, dtype=torch.float32, device=device)


def run_loro_subject(
    subject, X, y, groups, edge_index, model_factory, device,
    n_runs_per_subject, random_seed, batch_size, max_epochs, patience,
    learning_rate, weight_decay, n_classes=4,
):
    """Nested LORO CV for one subject. Returns (fold_df, cm_total, best_state, best_scaler) or None."""
    fold_rows, fold_cms = [], []
    best_overall_bal, best_state, best_scaler = -np.inf, None, None

    for test_run in range(1, n_runs_per_subject + 1):
        val_run = (test_run % n_runs_per_subject) + 1
        train_runs = [r for r in range(1, n_runs_per_subject + 1) if r != test_run and r != val_run]

        train_mask = np.isin(groups, train_runs)
        val_mask = groups == val_run
        test_mask = groups == test_run
        if not np.any(train_mask) or not np.any(val_mask) or not np.any(test_mask):
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_te)

        train_loader = DataLoader(make_graph_dataset(X_tr_s, y_tr, edge_index), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(make_graph_dataset(X_val_s, y_val, edge_index), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(make_graph_dataset(X_te_s, y_te, edge_index), batch_size=batch_size, shuffle=False)

        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        model = model_factory().to(device)

        class_w = class_weights_tensor(y_tr, n_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=50)

        best_fold_bal, best_fold_state, no_improve = -np.inf, None, 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch.x, batch.edge_index, batch.batch), batch.y)
                loss.backward()
                optimizer.step()

            _, val_bal, _, _ = evaluate_model(model, val_loader, device)
            scheduler.step(val_bal)

            if val_bal > best_fold_bal:
                best_fold_bal, best_fold_state, no_improve = val_bal, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        if best_fold_state is None:
            continue

        model.load_state_dict(best_fold_state)
        te_acc, te_bal, y_true, y_pred = evaluate_model(model, test_loader, device)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

        fold_rows.append({
            "subject": subject, "test_run": test_run, "val_run": val_run,
            "train_runs": train_runs, "accuracy": float(te_acc), "balanced_accuracy": float(te_bal),
        })
        fold_cms.append(cm)

        if te_bal > best_overall_bal:
            best_overall_bal = te_bal
            best_state = copy.deepcopy(best_fold_state)
            best_scaler = copy.deepcopy(scaler)

    if not fold_rows:
        return None

    fold_df = pd.DataFrame(fold_rows).sort_values("test_run").reset_index(drop=True)
    cm_total = np.sum(np.stack(fold_cms, axis=0), axis=0)
    return fold_df, cm_total, best_state, best_scaler