import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Build a support vector machine pipeline.
def build_svc_pipeline(random_seed=None):
    """Constructs a scaling and classification pipeline without a fixed C."""
    clf = LinearSVC(
        class_weight="balanced",
        dual="auto",
        max_iter=20000,
        random_state=random_seed,
    )
    return Pipeline(steps=[("scale", StandardScaler()), ("clf", clf)])


# Evaluate a pipeline using Leave-One-Run-Out cross-validation.
def evaluate_nested_loro_cv(X, y, groups, random_seed=None):
    """
    Runs nested Leave-One-Run-Out CV.
    Inner loop optimizes C; Outer loop evaluates generalizability.
    """
    logo_outer = LeaveOneGroupOut()
    
    # Nested CV requires at least 3 runs (1 test, 1 validation, 1+ train)
    if logo_outer.get_n_splits(groups=groups) < 3:
        raise ValueError("Not enough run groups for nested Leave-One-Run-Out CV. Minimum 3 required.")
        
    pipeline = build_svc_pipeline(random_seed=random_seed)
    
    # Define the logarithmic search space for C
    param_grid = {
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    y_pred = np.empty_like(y, dtype=object)
    fold_acc, fold_bacc, best_c_params = [], [], []
    
    # Outer Loop (Evaluation)
    for train_idx, test_idx in logo_outer.split(X, y, groups):
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Inner Loop (Hyperparameter Tuning)
        logo_inner = LeaveOneGroupOut()
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=logo_inner,
            scoring="balanced_accuracy", # Optimize inner folds for balanced accuracy
            n_jobs=-1,                   # Utilize all cores
            refit=True                   # Automatically retrain on all X_train with best C
        )
        
        # Fit GridSearch. We MUST pass groups_train so the inner CV splits correctly by run.
        grid_search.fit(X_train, y_train, groups=groups_train)
        
        # Predict on the outer hold-out run using the optimized pipeline
        best_model = grid_search.best_estimator_
        y_pred[test_idx] = best_model.predict(X_test)
        
        # Log metrics for this fold
        fold_acc.append(accuracy_score(y_test, y_pred[test_idx]))
        fold_bacc.append(balanced_accuracy_score(y_test, y_pred[test_idx]))
        
        # Store the best C found for this fold (useful for debugging stability)
        best_c_params.append(grid_search.best_params_["clf__C"])
        
    return {
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc, ddof=0)),
        "mean_balanced_accuracy": float(np.mean(fold_bacc)),
        "std_balanced_accuracy": float(np.std(fold_bacc, ddof=0)),
        "best_C_per_fold": best_c_params
    }