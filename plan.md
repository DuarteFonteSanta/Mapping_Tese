### Main problems
1. Features: There is no single-trial GLM, detrending, high-pass filtering, confound regression, run-wise normalisation or dimensionality reduction. THere is 80 trainig trials vs. 2187 voxels (most likely the cause of the results)

2. Run-level drift: evaluation is leave-one-run-out, drift between runs is not removed by fitting the scaler on trainig trials only.

3. Tune the hyperparameters.

4. CNN problem: low learning rate?; vfew batches per epoch?; information lost (ROI into a bounding box)?; unused mask channel?

5. Explore the ANN, results dipped.

### Plan

1. Phase 0:
    - rework the project, delete files, rename files

2. Phase 1:
    - Single-trial estimates with a nilearn GLM (least-squares-separate betas) and compare them against the current average of three volumes;
    - Add ablations: high-pass or detrend, fMRIPrep confound regression, run-wise z-scoring, optional light smoothing, and PCA or ANOVA feature selection fitted inside each fold;

3. Phase 2:
    - ROI choice: justify Destrieux G_postcentral against alternatives such as Juelich/BA 3a/3b/1/2 or HCP-MMP.
    - Evaluation scheme: keep within-subject leave-one-run-out as primary. Add leave-one-subject-out on all 10 subjects as a secondary experiment, since pooling probably explains the higher accuracy, but only with strict subject-level splits;
    - Tune each model with nested inner-CV search (Optuna or random search) on the cluster:
        - SVC: C
        - ANN: width, dropout, weight decay, learning rate
        - GNN/GCN: hidden size, heads, pooling, learning rate, patience
        - CNN: Overfit one batch; Log the prediction distribution per epoch; Run a label-shuffle sanity check; Try a learning rate around 1e-3, remove the mask channel or use masked convolution, replace max-pool with smaller strides, and use a larger batch or batch-norm alternatives
    - Add per-subject label-permutation tests and run 3–5 seeds per neural model. Replace the "chance = 25%" assumption with permutation-derived thresholds
    - Rerun SearchLight with the final preprocessing, on the cluster. It is the independent reference map for the SHAP comparison

4. Phase 3:
    - Final runs on all subjects for all models
    - Rerun SHAP only for models that beat their permutation threshold. SHAP on a near-chance model is not interpretable, and the chapter should say so. Check stability: seed-to-seed and bootstrap SHAP agreement, background size, and GradientExplainer sample count
    - Gradient measure: replace the raw MNI-z centroid with a position along the gyrus (first principal axis of the ROI voxels, or surface arc length), computed per subject and tested across subjects (N=10) with a one-sample test.
    - Group averaging: check that voxel index equals voxel location across subjects, and normalise per subject before averaging.
    - urface projection: document vol_to_surf settings. Use nearest-neighbour interpolation for categorical maps, which fixes the all-red winner-take-all bug.
