import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def build_svc_pipeline(random_seed=None):
    clf = LinearSVC(
        class_weight="balanced",
        max_iter=20000,
        random_state=random_seed,
        dual="auto"
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("clf", clf)])

def evaluate_nested_loro_cv(X, y, groups, random_seed=None):
    logo_outer = LeaveOneGroupOut()
    
    if logo_outer.get_n_splits(groups=groups) < 3:
        raise ValueError("Not enough run groups for nested LORO CV. Minimum 3 required.")
        
    pipeline = build_svc_pipeline(random_seed=random_seed)
    param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    
    y_pred = np.empty_like(y, dtype=object)
    fold_acc, fold_bacc, best_c_params, fold_weights = [], [], [], []
    
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
        
        fold_weights.append(best_model.named_steps["clf"].coef_)
        fold_acc.append(accuracy_score(y[test_idx], y_pred[test_idx]))
        fold_bacc.append(balanced_accuracy_score(y[test_idx], y_pred[test_idx]))
        best_c_params.append(grid_search.best_params_["clf__C"])
        
    return {
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc, ddof=0)),
        "mean_balanced_accuracy": float(np.mean(fold_bacc)),
        "std_balanced_accuracy": float(np.std(fold_bacc, ddof=0)),
        "best_C_per_fold": best_c_params,
        "mean_weights": np.mean(fold_weights, axis=0),
        # Return arrays for later visualization
        "y_true": y,
        "y_pred": y_pred
    }