import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def build_rbf_svc_pipeline(random_seed=None):
    clf = SVC(
        kernel="rbf",
        class_weight="balanced",
        max_iter=20000,
        random_state=random_seed,
        decision_function_shape="ovo"  # One-vs-One strategy for multi-class classification
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("clf", clf)])

def evaluate_nested_loro_rbf_cv(X, y, groups, random_seed=None):
    logo_outer = LeaveOneGroupOut()
    
    if logo_outer.get_n_splits(groups=groups) < 3:
        raise ValueError("Not enough run groups for nested LORO CV. Minimum 3 required.")
        
    pipeline = build_rbf_svc_pipeline(random_seed=random_seed)
    
    # Expand the parameter grid to include both C and gamma.
    # 'scale' and 'auto' use variance-based heuristics, while floats explicitly set the radius.
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", "auto", 0.01, 0.001, 0.0001]
    }
    
    y_pred = np.empty_like(y, dtype=object)
    fold_acc, fold_bacc, best_params_list = [], [], []
    
    for train_idx, test_idx in logo_outer.split(X, y, groups):
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        
        logo_inner = LeaveOneGroupOut()

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=logo_inner,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True
        )
        
        grid_search.fit(X_train, y_train, groups=groups_train)
        
        best_model = grid_search.best_estimator_
        y_pred[test_idx] = best_model.predict(X[test_idx])
        
        fold_acc.append(accuracy_score(y[test_idx], y_pred[test_idx]))
        fold_bacc.append(balanced_accuracy_score(y[test_idx], y_pred[test_idx]))
        
        # Store the dictionary of both best C and best gamma for this fold
        best_params_list.append(grid_search.best_params_)
        
    return {
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc, ddof=0)),
        "mean_balanced_accuracy": float(np.mean(fold_bacc)),
        "std_balanced_accuracy": float(np.std(fold_bacc, ddof=0)),
        "best_params_per_fold": best_params_list,
        "y_true": y,
        "y_pred": y_pred
    }