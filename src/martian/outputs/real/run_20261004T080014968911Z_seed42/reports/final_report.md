# Final Report -- run `run_20261004T080014968911Z_seed42`

[DATA SOURCE: REAL (Martian Chapter 4 mars.run_all pipeline)]

- Data source: `real`
- Started (UTC): 2026-10-04T08:00:14.975027+00:00
- Finished (UTC): 2026-10-04T09:13:31.779684+00:00
- Status: `completed`
- Seed strategy: {'global_generation_seed': 42, 'task_seeds': {'dust_regression': 42, 'storm_detection': 44}, 'bootstrap_seed': 'see configs/evaluation.yaml bootstrap.seed', 'note': 'see planning/SEED_STRATEGY.md for the full rationale'}

## Pipeline summary

- **n_retrievals_generated**: 1077710
- **n_qc_accepted**: 268673
- **n_qc_rejected**: 809037
- **dust_harmonisation**: {'enabled': True, 'n_retrievals_harmonised': 268673, 'extinction_to_absorption_ratio': 1.3, 'reference_pressure_pa': 610.0, 'factor_p05_p50_p95': [0.5436310477661382, 0.8259993277509219, 1.0717752366027042], 'pressure_mbar_min_median_max': [3.787557905081516, 5.680764541400022, 13.78133077739962]}
- **n_aggregated_cell_sols**: 142608
- **n_trailing_filled_rows**: 232702
- **n_atmospheric_anomaly**: 11712
- **n_storm_episode**: 5629
- **n_candidate_episodes**: 3886
- **n_episodes_excluded_for_split_boundary_crossing**: 0
- **catalogue_validation**: {'covered_mars_year': 36, 'n_documented_events': 98, 'n_qualifying_episodes': 109, 'n_hits': 7, 'n_misses': 91, 'pod': 0.07142857142857142, 'n_matched_episodes': 2, 'n_false_positive_candidates': 107, 'episode_precision': 0.01834862385321101, 'mean_temporal_offset_sols': 6.0}
- **catalogue_tolerance_tuning**: {'selected_tolerance': {'temporal_tolerance_sols': 3, 'spatial_region_radius_cells': 3, 'dev_f1': 0.05063291139240506, 'dev_pod': 0.08163265306122448, 'dev_episode_precision': 0.03669724770642202}, 'split': {'dev_fraction': 0.5, 'seed': 42, 'n_dev_events': 49, 'n_test_events': 49}, 'test_validation_summary': {'covered_mars_year': 36, 'n_documented_events': 49, 'n_qualifying_episodes': 109, 'n_hits': 5, 'n_misses': 44, 'pod': 0.10204081632653061, 'n_matched_episodes': 4, 'n_false_positive_candidates': 105, 'episode_precision': 0.03669724770642202, 'mean_temporal_offset_sols': 7.428571428571429}}
- **n_engineered_feature_columns**: 32
- **split_sizes**: {'train': 81181, 'test': 48273, 'val': 11139, 'unassigned': 2015}

## Step timings (seconds)

- 3_qc_and_gridding: 1350.31s
- 4_aggregate_cell_sols: 146.70s
- 5_causal_missing_data: 24.22s
- 6_causal_filtering: 7.92s
- 7_climatological_deviations: 9.98s
- 8_build_labels_and_episodes: 39.07s
- 9_validate_labels_against_documented_catalogue: 9.46s
- 10_build_causal_features: 176.07s
- 11_chronological_splits: 0.06s
- 12_leakage_checks: 0.01s
- 13_train_evaluate_uncertainty_figures: 2630.50s

## Leakage checks

```
{'episode_boundary_check': 'passed', 'feature_manifest_target_leakage_check': 'passed', 'tasks_checked': ['dust_regression', 'storm_detection']}
```

## Task results

### Task: dust_regression

**Horizon 1 sol** -- status: `trained`, n_train=32956, n_val=3077, n_test=13456

- `[baseline]` **climatology**: mae=nan (90% CI [0.2126, 0.2214]), rmse=nan (90% CI [0.3093, 0.3607]), r2=nan (90% CI [-0.5175, -0.3339]), correlation=nan (90% CI [-0.01137, 0.02946])
- `[baseline]` **persistence**: mae=0.1872 (90% CI [0.1804, 0.1932]), rmse=0.4148 (90% CI [0.383, 0.4529]), r2=-1.039 (90% CI [-1.288, -0.8142]), correlation=0.01589 (90% CI [0.001033, 0.03573])
- `[baseline]` **seasonal_mean**: mae=0.1218 (90% CI [0.1176, 0.1266]), rmse=0.2929 (90% CI [0.2634, 0.3269]), r2=-0.01726 (90% CI [-0.02019, -0.01462]), correlation=-0.04285 (90% CI [-0.06031, -0.02539])
- `[classical]` **random_forest_regressor**: mae=0.1227 (90% CI [0.1184, 0.1272]), rmse=0.2947 (90% CI [0.2643, 0.329]), r2=-0.02965 (90% CI [-0.03747, -0.02227]), correlation=0.07759 (90% CI [0.05872, 0.1008])
- `[classical]` **gradient_boosting_regressor**: mae=0.1259 (90% CI [0.1215, 0.1305]), rmse=0.2987 (90% CI [0.2681, 0.3328]), r2=-0.05769 (90% CI [-0.06871, -0.04633]), correlation=0.0561 (90% CI [0.03714, 0.07888])
- `[sequence]` lstm window=3 sol (n_train_windows=22213): {'mae': 0.16907398942364396, 'rmse': 0.30806097656187076, 'mse': 0.0949015652802535, 'r2': -0.5320816712446503, 'bias': -0.15802608268182353, 'correlation': 0.035768599591932945}
- `[sequence]` gru window=3 sol (n_train_windows=22213): {'mae': 0.11690346147851716, 'rmse': 0.26106601570880494, 'mse': 0.06815546455806999, 'r2': -0.10029521363764471, 'bias': -0.021580434389802904, 'correlation': 0.04067994379388369}
- `[sequence]` lstm window=5 sol (n_train_windows=15057): {'mae': 0.12529584128291288, 'rmse': 0.2387852912121739, 'mse': 0.0570184152992827, 'r2': -0.2781594162896974, 'bias': -0.08453889432339526, 'correlation': 0.0731064230698552}
- `[sequence]` gru window=5 sol (n_train_windows=15057): {'mae': 0.1107163735556547, 'rmse': 0.21955455908873905, 'mse': 0.0482042044166506, 'r2': -0.08057471356399626, 'bias': -0.011740402148294294, 'correlation': 0.09550685349668987}
- `[sequence]` lstm window=10 sol (n_train_windows=8518): {'mae': 0.11144559753828823, 'rmse': 0.23593999218338402, 'mse': 0.055667679911495314, 'r2': -0.11487776541309613, 'bias': -0.02800255037470334, 'correlation': 0.014332171236511421}
- `[sequence]` gru window=10 sol (n_train_windows=8518): {'mae': 0.10561057098888275, 'rmse': 0.23006782742930199, 'mse': 0.052931205218039075, 'r2': -0.06007334754979676, 'bias': -0.004841226117248089, 'correlation': 0.06616144400660053}

**Horizon 3 sol** -- status: `trained`, n_train=28878, n_val=3381, n_test=15548

- `[baseline]` **climatology**: mae=nan (90% CI [0.2025, 0.2067]), rmse=nan (90% CI [0.275, 0.3025]), r2=nan (90% CI [-0.8772, -0.5981]), correlation=nan (90% CI [0.07906, 0.1147])
- `[baseline]` **persistence**: mae=0.1369 (90% CI [0.1315, 0.1414]), rmse=0.3773 (90% CI [0.3458, 0.4082]), r2=-2.305 (90% CI [-3.279, -1.654]), correlation=0.03568 (90% CI [0.02451, 0.05044])
- `[baseline]` **seasonal_mean**: mae=0.1001 (90% CI [0.09715, 0.1026]), rmse=0.2074 (90% CI [0.1787, 0.2364]), r2=0.001065 (90% CI [-0.0003758, 0.002794]), correlation=0.04529 (90% CI [0.02392, 0.06902])
- `[classical]` **random_forest_regressor**: mae=0.09523 (90% CI [0.09224, 0.09788]), rmse=0.2113 (90% CI [0.1832, 0.2398]), r2=-0.03672 (90% CI [-0.04976, -0.02556]), correlation=0.1163 (90% CI [0.09745, 0.1443])
- `[classical]` **gradient_boosting_regressor**: mae=0.09834 (90% CI [0.09524, 0.101]), rmse=0.212 (90% CI [0.1846, 0.2404]), r2=-0.04367 (90% CI [-0.05974, -0.03077]), correlation=0.1216 (90% CI [0.1018, 0.1503])
- `[sequence]` lstm window=3 sol (n_train_windows=19460): {'mae': 0.14004659012675605, 'rmse': 0.2237150216270672, 'mse': 0.05004841090159914, 'r2': -0.7098498876571335, 'bias': -0.12757086906403192, 'correlation': 0.09730976483448613}
- `[sequence]` gru window=3 sol (n_train_windows=19460): {'mae': 0.09745829427650066, 'rmse': 0.1758919748948463, 'mse': 0.03093798683240924, 'r2': -0.056962895659941726, 'bias': 0.007442841825756655, 'correlation': 0.14410986724712155}
- `[sequence]` lstm window=5 sol (n_train_windows=14364): {'mae': 0.10818512979988514, 'rmse': 0.1963592589923583, 'mse': 0.03855695859202804, 'r2': -0.34249020856559476, 'bias': -0.06601552390070473, 'correlation': 0.0561426746482162}
- `[sequence]` gru window=5 sol (n_train_windows=14364): {'mae': 0.09768981929777373, 'rmse': 0.17279167651385416, 'mse': 0.029856963472468415, 'r2': -0.039570614046744, 'bias': 0.021473762644791273, 'correlation': 0.1790911596578441}
- `[sequence]` lstm window=10 sol (n_train_windows=8229): {'mae': 0.1023781194000958, 'rmse': 0.19049112091552647, 'mse': 0.03628686714765372, 'r2': -0.1339597169564235, 'bias': -0.01407602576808282, 'correlation': 0.04482651342682513}
- `[sequence]` gru window=10 sol (n_train_windows=8229): {'mae': 0.10324597674487522, 'rmse': 0.18775975051735921, 'mse': 0.03525372391434097, 'r2': -0.10167412989662883, 'bias': 0.028194147567898575, 'correlation': 0.08044475917693476}

**Horizon 5 sol** -- status: `trained`, n_train=28397, n_val=4653, n_test=19560

- `[baseline]` **climatology**: mae=nan (90% CI [0.2061, 0.2125]), rmse=nan (90% CI [0.2613, 0.2954]), r2=nan (90% CI [-1.09, -0.7141]), correlation=nan (90% CI [0.06698, 0.118])
- `[baseline]` **persistence**: mae=0.1144 (90% CI [0.1109, 0.1182]), rmse=0.339 (90% CI [0.3102, 0.3693]), r2=-1.841 (90% CI [-2.508, -1.225]), correlation=0.08479 (90% CI [0.06807, 0.1083])
- `[baseline]` **seasonal_mean**: mae=0.1005 (90% CI [0.09806, 0.103]), rmse=0.2016 (90% CI [0.1808, 0.2234]), r2=-0.004587 (90% CI [-0.006658, -0.002938]), correlation=-0.001644 (90% CI [-0.02192, 0.02197])
- `[classical]` **random_forest_regressor**: mae=0.09421 (90% CI [0.0918, 0.09673]), rmse=0.2022 (90% CI [0.1817, 0.2245]), r2=-0.01153 (90% CI [-0.02047, -0.0026]), correlation=0.1821 (90% CI [0.1553, 0.2089])
- `[classical]` **gradient_boosting_regressor**: mae=0.1005 (90% CI [0.09812, 0.1029]), rmse=0.2175 (90% CI [0.1984, 0.2384]), r2=-0.1695 (90% CI [-0.2197, -0.1206]), correlation=0.1268 (90% CI [0.1092, 0.1438])
- `[sequence]` lstm window=3 sol (n_train_windows=20813): {'mae': 0.14798196713072898, 'rmse': 0.24828259459351457, 'mse': 0.061644246778087504, 'r2': -0.7625939138621245, 'bias': -0.13939284427278872, 'correlation': -0.029737210534650838}
- `[sequence]` gru window=3 sol (n_train_windows=20813): {'mae': 0.0949996070056219, 'rmse': 0.18395940977947906, 'mse': 0.0338410644464143, 'r2': 0.03238246309356019, 'bias': -0.0025250573427839595, 'correlation': 0.23454299827672415}
- `[sequence]` lstm window=5 sol (n_train_windows=16021): {'mae': 0.11755207015878023, 'rmse': 0.22604861877037774, 'mse': 0.05109797804799557, 'r2': -0.4339250432377679, 'bias': -0.08021621849382096, 'correlation': -0.06869438879489825}
- `[sequence]` gru window=5 sol (n_train_windows=16021): {'mae': 0.09689212597339229, 'rmse': 0.18540819085054408, 'mse': 0.03437619723447178, 'r2': 0.03532601506286792, 'bias': 0.003697184350603223, 'correlation': 0.2421920913136321}
- `[sequence]` lstm window=10 sol (n_train_windows=8517): {'mae': 0.10325345024654033, 'rmse': 0.1796758522820031, 'mse': 0.032283411893264204, 'r2': -0.20717872156152417, 'bias': -0.020216439569306285, 'correlation': 0.006055701934132052}
- `[sequence]` gru window=10 sol (n_train_windows=8517): {'mae': 0.09811720210817627, 'rmse': 0.1689376043778929, 'mse': 0.028539914172941456, 'r2': -0.06719751984936617, 'bias': 0.014202920404591994, 'correlation': 0.13253844728089315}

### Task: storm_detection

**Horizon 1 sol** -- status: `trained`, n_train=32956, n_val=3077, n_test=13456

- `[baseline]` **climatology**: tss=0 (90% CI [0, 0]), precision=0.03656 (90% CI [0.03337, 0.04002]), recall=1 (90% CI [1, 1]), pr_auc=0.03656 (90% CI [0.03337, 0.04002]), roc_auc=0.5 (90% CI [0.5, 0.5])
- `[baseline]` **persistence**: tss=0.3008 (90% CI [0.2641, 0.3396]), precision=0.1567 (90% CI [0.1392, 0.1783]), recall=0.378 (90% CI [0.3406, 0.4173]), pr_auc=0.08198 (90% CI [0.07146, 0.09517]), roc_auc=0.6504 (90% CI [0.6321, 0.6698])
- `[baseline]` **seasonal_mean**: tss=0 (90% CI [0, 0]), precision=nan (90% CI [nan, nan]), recall=0 (90% CI [0, 0]), pr_auc=0.1307 (90% CI [0.1189, 0.1434]), roc_auc=0.8531 (90% CI [0.841, 0.8637])
- `[classical]` **logistic_regression**: tss=0.0009796 (90% CI [-0.003142, 0.006585]), precision=0.04762 (90% CI [0, 0.1112]), recall=0.004065 (90% CI [0, 0.009259]), pr_auc=0.04353 (90% CI [0.03767, 0.05209]), roc_auc=0.5272 (90% CI [0.5095, 0.5488])
- `[classical]` **random_forest_classifier**: tss=0.6497 (90% CI [0.6322, 0.6642]), precision=0.1116 (90% CI [0.1017, 0.1218]), recall=0.9309 (90% CI [0.914, 0.9456]), pr_auc=0.1534 (90% CI [0.1394, 0.1745]), roc_auc=0.8781 (90% CI [0.8712, 0.8851])
- `[classical]` **gradient_boosting_classifier**: tss=0.2873 (90% CI [0.2537, 0.3232]), precision=0.1476 (90% CI [0.1313, 0.1656]), recall=0.3679 (90% CI [0.3319, 0.4048]), pr_auc=0.1265 (90% CI [0.1132, 0.146]), roc_auc=0.7262 (90% CI [0.7016, 0.7515])
- `[classical]` **knn_classifier**: tss=0.4374 (90% CI [0.4039, 0.4809]), precision=0.1283 (90% CI [0.1152, 0.1414]), recall=0.5894 (90% CI [0.5565, 0.6318]), pr_auc=0.1005 (90% CI [0.08945, 0.1159]), roc_auc=0.7236 (90% CI [0.707, 0.7455])
- `[sequence]` lstm window=3 sol (n_train_windows=22213): {'tss': 0.6975541529511855, 'hss': 0.11502939006524818, 'pod': 0.8695652173913043, 'far': 0.17201106444011885, 'precision': 0.07696536558548653, 'recall': 0.8695652173913043, 'f1': 0.1414141414141414, 'brier': 0.12298448146742068, 'log_score': 0.4302384645094824, 'pr_auc': 0.07994159596635483, 'roc_auc': 0.8665503038139485, 'tp': 140, 'fp': 1679, 'tn': 8082, 'fn': 21, 'n_classes_present_in_y_true': 2, 'prevalence': 0.01622656722434993}
- `[sequence]` gru window=3 sol (n_train_windows=22213): {'tss': 0.414591978090016, 'hss': 0.08460326749832914, 'pod': 0.5527950310559007, 'far': 0.13820305296588464, 'precision': 0.061891515994436715, 'recall': 0.5527950310559007, 'f1': 0.11131957473420886, 'brier': 0.10733164981331736, 'log_score': 0.3932133495912719, 'pr_auc': 0.04806051171810884, 'roc_auc': 0.7125345445590611, 'tp': 89, 'fp': 1349, 'tn': 8412, 'fn': 72, 'n_classes_present_in_y_true': 2, 'prevalence': 0.01622656722434993}
- `[sequence]` lstm window=5 sol (n_train_windows=15057): {'tss': 0.643065793065793, 'hss': 0.04779210924490374, 'pod': 0.85, 'far': 0.20693420693420694, 'precision': 0.03254626675175495, 'recall': 0.85, 'f1': 0.0626920712968654, 'brier': 0.10301373659223655, 'log_score': 0.38555715601851626, 'pr_auc': 0.06164484204409029, 'roc_auc': 0.8433422058422059, 'tp': 51, 'fp': 1516, 'tn': 5810, 'fn': 9, 'n_classes_present_in_y_true': 2, 'prevalence': 0.008123476848090982}
- `[sequence]` gru window=5 sol (n_train_windows=15057): {'tss': 0.2822959322959323, 'hss': 0.05909304561631095, 'pod': 0.35, 'far': 0.0677040677040677, 'precision': 0.04061895551257253, 'recall': 0.35, 'f1': 0.07279029462738301, 'brier': 0.08957718689837686, 'log_score': 0.3522000588730144, 'pr_auc': 0.033398265325027925, 'roc_auc': 0.6949779324779325, 'tp': 21, 'fp': 496, 'tn': 6830, 'fn': 39, 'n_classes_present_in_y_true': 2, 'prevalence': 0.008123476848090982}
- `[sequence]` lstm window=10 sol (n_train_windows=8518): {'tss': 0.6458289433635364, 'hss': 0.0544219566078414, 'pod': 0.782608695652174, 'far': 0.1367797522886376, 'precision': 0.034220532319391636, 'recall': 0.782608695652174, 'f1': 0.06557377049180328, 'brier': 0.09239992438823035, 'log_score': 0.36121778062722126, 'pr_auc': 0.09650863248840592, 'roc_auc': 0.7872445037578144, 'tp': 18, 'fp': 508, 'tn': 3206, 'fn': 5, 'n_classes_present_in_y_true': 2, 'prevalence': 0.006154669521006155}
- `[sequence]` gru window=10 sol (n_train_windows=8518): {'tss': 0.5551146074781672, 'hss': 0.10874246388743014, 'pod': 0.6086956521739131, 'far': 0.05358104469574582, 'precision': 0.06572769953051644, 'recall': 0.6086956521739131, 'f1': 0.11864406779661017, 'brier': 0.07710952131418804, 'log_score': 0.32248882674197876, 'pr_auc': 0.08309546760648184, 'roc_auc': 0.7915525274519445, 'tp': 14, 'fp': 199, 'tn': 3515, 'fn': 9, 'n_classes_present_in_y_true': 2, 'prevalence': 0.006154669521006155}

**Horizon 3 sol** -- status: `trained`, n_train=60733, n_val=8522, n_test=37943

- `[baseline]` **climatology**: tss=0 (90% CI [0, 0]), precision=0.03474 (90% CI [0.03261, 0.03725]), recall=1 (90% CI [1, 1]), pr_auc=0.03474 (90% CI [0.03261, 0.03725]), roc_auc=0.5 (90% CI [0.5, 0.5])
- `[baseline]` **persistence**: tss=0.2492 (90% CI [0.2304, 0.2693]), precision=0.1966 (90% CI [0.1818, 0.2126]), recall=0.2921 (90% CI [0.2718, 0.312]), pr_auc=0.08203 (90% CI [0.07353, 0.09093]), roc_auc=0.6246 (90% CI [0.6152, 0.6346])
- `[baseline]` **seasonal_mean**: tss=0 (90% CI [0, 0]), precision=nan (90% CI [nan, nan]), recall=0 (90% CI [0, 0]), pr_auc=0.1767 (90% CI [0.1668, 0.1905]), roc_auc=0.8928 (90% CI [0.8841, 0.9011])
- `[classical]` **logistic_regression**: tss=-0.0001911 (90% CI [-0.0003251, -8.194e-05]), precision=0 (90% CI [0, 0]), recall=0 (90% CI [0, 0]), pr_auc=0.02435 (90% CI [0.02287, 0.02628]), roc_auc=0.3564 (90% CI [0.3464, 0.3658])
- `[classical]` **random_forest_classifier**: tss=0.599 (90% CI [0.5824, 0.6132]), precision=0.09304 (90% CI [0.08696, 0.1006]), recall=0.9226 (90% CI [0.9085, 0.9356]), pr_auc=0.1598 (90% CI [0.1503, 0.172]), roc_auc=0.8742 (90% CI [0.8674, 0.8813])
- `[classical]` **gradient_boosting_classifier**: tss=0.1848 (90% CI [0.1675, 0.2043]), precision=0.1985 (90% CI [0.1814, 0.2161]), recall=0.2162 (90% CI [0.1976, 0.2361]), pr_auc=0.1216 (90% CI [0.1101, 0.1343]), roc_auc=0.5782 (90% CI [0.5568, 0.5945])
- `[classical]` **knn_classifier**: tss=0.4299 (90% CI [0.4074, 0.451]), precision=0.142 (90% CI [0.1328, 0.1535]), recall=0.5493 (90% CI [0.5269, 0.5689]), pr_auc=0.109 (90% CI [0.09922, 0.1217]), roc_auc=0.7182 (90% CI [0.7074, 0.7289])
- `[sequence]` lstm window=3 sol (n_train_windows=39107): {'tss': 0.7565197963741714, 'hss': 0.1811232308855718, 'pod': 0.8742004264392325, 'far': 0.11768063006506106, 'precision': 0.11704253497002569, 'recall': 0.8742004264392325, 'f1': 0.20644511581067473, 'brier': 0.1189829930896389, 'log_score': 0.4215803402045735, 'pr_auc': 0.11199377869104152, 'roc_auc': 0.9045701263603876, 'tp': 410, 'fp': 3093, 'tn': 23190, 'fn': 59, 'n_classes_present_in_y_true': 2, 'prevalence': 0.0175313995215311}
- `[sequence]` gru window=3 sol (n_train_windows=39107): {'tss': 0.40497992695060087, 'hss': 0.09561651946795667, 'pod': 0.5309168443496801, 'far': 0.12593691739907925, 'precision': 0.06996347288564203, 'recall': 0.5309168443496801, 'f1': 0.12363455809334657, 'brier': 0.10613067744203722, 'log_score': 0.3908229761122856, 'pr_auc': 0.06555495699882274, 'roc_auc': 0.7308513443998557, 'tp': 249, 'fp': 3310, 'tn': 22973, 'fn': 220, 'n_classes_present_in_y_true': 2, 'prevalence': 0.0175313995215311}
- `[sequence]` lstm window=5 sol (n_train_windows=26735): {'tss': 0.7828786188797584, 'hss': 0.21418949870337842, 'pod': 0.8516949152542372, 'far': 0.06881629637447886, 'precision': 0.1335548172757475, 'recall': 0.8516949152542372, 'f1': 0.2309017805858702, 'brier': 0.10190332713067894, 'log_score': 0.3829951645545527, 'pr_auc': 0.1145184124579822, 'roc_auc': 0.9200322274508471, 'tp': 201, 'fp': 1304, 'tn': 17645, 'fn': 35, 'n_classes_present_in_y_true': 2, 'prevalence': 0.012301277039353662}
- `[sequence]` gru window=5 sol (n_train_windows=26735): {'tss': 0.5358363797204092, 'hss': 0.11636071271732311, 'pod': 0.6313559322033898, 'far': 0.09551955248298064, 'precision': 0.07605921388463502, 'recall': 0.6313559322033898, 'f1': 0.1357630979498861, 'brier': 0.09054771676500138, 'log_score': 0.35435994556060985, 'pr_auc': 0.09047710240381972, 'roc_auc': 0.7981941491478911, 'tp': 149, 'fp': 1810, 'tn': 17139, 'fn': 87, 'n_classes_present_in_y_true': 2, 'prevalence': 0.012301277039353662}
- `[sequence]` lstm window=10 sol (n_train_windows=14601): {'tss': 0.7951818190088885, 'hss': 0.09868990628557621, 'pod': 0.8985507246376812, 'far': 0.10336890562879264, 'precision': 0.05904761904761905, 'recall': 0.8985507246376812, 'f1': 0.11081322609472745, 'brier': 0.09108805369092317, 'log_score': 0.3581831299328768, 'pr_auc': 0.16908305847993632, 'roc_auc': 0.9032899066265152, 'tp': 62, 'fp': 988, 'tn': 8570, 'fn': 7, 'n_classes_present_in_y_true': 2, 'prevalence': 0.007167341851043939}
- `[sequence]` gru window=10 sol (n_train_windows=14601): {'tss': 0.5959405733417033, 'hss': 0.10459273795490572, 'pod': 0.6666666666666666, 'far': 0.07072609332496338, 'precision': 0.06371191135734072, 'recall': 0.6666666666666666, 'f1': 0.11630847029077118, 'brier': 0.07927589479804403, 'log_score': 0.32775803933433806, 'pr_auc': 0.08730108894241072, 'roc_auc': 0.8350603940549081, 'tp': 46, 'fp': 676, 'tn': 8882, 'fn': 23, 'n_classes_present_in_y_true': 2, 'prevalence': 0.007167341851043939}

**Horizon 5 sol** -- status: `trained`, n_train=66808, n_val=9766, n_test=42131

- `[baseline]` **climatology**: tss=0 (90% CI [0, 0]), precision=0.0455 (90% CI [0.04316, 0.04798]), recall=1 (90% CI [1, 1]), pr_auc=0.0455 (90% CI [0.04316, 0.04798]), roc_auc=0.5 (90% CI [0.5, 0.5])
- `[baseline]` **persistence**: tss=0.1797 (90% CI [0.1641, 0.1952]), precision=0.1965 (90% CI [0.1833, 0.2108]), recall=0.2233 (90% CI [0.2071, 0.2387]), pr_auc=0.07922 (90% CI [0.07326, 0.08583]), roc_auc=0.5899 (90% CI [0.5821, 0.5976])
- `[baseline]` **seasonal_mean**: tss=0 (90% CI [0, 0]), precision=nan (90% CI [nan, nan]), recall=0 (90% CI [0, 0]), pr_auc=0.2223 (90% CI [0.2106, 0.2357]), roc_auc=0.8944 (90% CI [0.8868, 0.903])
- `[classical]` **logistic_regression**: tss=-0.0001741 (90% CI [-0.0002968, -7.676e-05]), precision=0 (90% CI [0, 0]), recall=0 (90% CI [0, 0]), pr_auc=0.02979 (90% CI [0.02813, 0.03147]), roc_auc=0.3016 (90% CI [0.295, 0.3101])
- `[classical]` **random_forest_classifier**: tss=0.4664 (90% CI [0.4563, 0.4763]), precision=0.08347 (90% CI [0.07919, 0.08851]), recall=0.9786 (90% CI [0.9723, 0.9845]), pr_auc=0.184 (90% CI [0.1725, 0.1965]), roc_auc=0.8647 (90% CI [0.8585, 0.8709])
- `[classical]` **gradient_boosting_classifier**: tss=0.1559 (90% CI [0.1402, 0.1697]), precision=0.1933 (90% CI [0.1752, 0.2094]), recall=0.1946 (90% CI [0.1788, 0.2078]), pr_auc=0.1152 (90% CI [0.1046, 0.1259]), roc_auc=0.5513 (90% CI [0.5349, 0.5671])
- `[classical]` **knn_classifier**: tss=0.4768 (90% CI [0.4551, 0.4956]), precision=0.1763 (90% CI [0.1649, 0.1865]), recall=0.6135 (90% CI [0.5927, 0.6319]), pr_auc=0.1353 (90% CI [0.1245, 0.1465]), roc_auc=0.7414 (90% CI [0.7309, 0.7512])
- `[sequence]` lstm window=3 sol (n_train_windows=42693): {'tss': 0.750276133992512, 'hss': 0.2578176862453759, 'pod': 0.8689903846153846, 'far': 0.11871425062287258, 'precision': 0.1760837798343887, 'recall': 0.8689903846153846, 'f1': 0.2928311057108141, 'brier': 0.12199399669204583, 'log_score': 0.4278088345332546, 'pr_auc': 0.16323058294465287, 'roc_auc': 0.9086936614110527, 'tp': 723, 'fp': 3383, 'tn': 25114, 'fn': 109, 'n_classes_present_in_y_true': 2, 'prevalence': 0.02836782706536193}
- `[sequence]` gru window=3 sol (n_train_windows=42693): {'tss': 0.4727103949538548, 'hss': 0.18553653599132877, 'pod': 0.578125, 'far': 0.10541460504614521, 'precision': 0.13802008608321378, 'recall': 0.578125, 'f1': 0.22283993514014364, 'brier': 0.10893463236476218, 'log_score': 0.3968389144814602, 'pr_auc': 0.16046883152339853, 'roc_auc': 0.7788584906710828, 'tp': 481, 'fp': 3004, 'tn': 25493, 'fn': 351, 'n_classes_present_in_y_true': 2, 'prevalence': 0.02836782706536193}
- `[sequence]` lstm window=5 sol (n_train_windows=29152): {'tss': 0.8331343907979465, 'hss': 0.2805009844324484, 'pod': 0.9342629482071713, 'far': 0.10112855740922473, 'precision': 0.18537549407114626, 'recall': 0.9342629482071713, 'f1': 0.3093667546174143, 'brier': 0.10533297404363615, 'log_score': 0.3900914900465141, 'pr_auc': 0.20398833693908103, 'roc_auc': 0.9322509764670465, 'tp': 469, 'fp': 2061, 'tn': 18319, 'fn': 33, 'n_classes_present_in_y_true': 2, 'prevalence': 0.024039842926922707}
- `[sequence]` gru window=5 sol (n_train_windows=29152): {'tss': 0.5810109903858559, 'hss': 0.23561187886293516, 'pod': 0.6633466135458167, 'far': 0.08233562315996075, 'precision': 0.165589259075087, 'recall': 0.6633466135458167, 'f1': 0.2650218861918026, 'brier': 0.09407924901108312, 'log_score': 0.3620075783969064, 'pr_auc': 0.219219539171949, 'roc_auc': 0.8408257060081558, 'tp': 333, 'fp': 1678, 'tn': 18702, 'fn': 169, 'n_classes_present_in_y_true': 2, 'prevalence': 0.024039842926922707}
- `[sequence]` lstm window=10 sol (n_train_windows=15371): {'tss': 0.8097052570893236, 'hss': 0.1481771980832098, 'pod': 0.9090909090909091, 'far': 0.09938565200158542, 'precision': 0.09066183136899365, 'recall': 0.9090909090909091, 'f1': 0.1648804616652927, 'brier': 0.09198214216727989, 'log_score': 0.3599503895531356, 'pr_auc': 0.22343396791869827, 'roc_auc': 0.9202969048391165, 'tp': 100, 'fp': 1003, 'tn': 9089, 'fn': 10, 'n_classes_present_in_y_true': 2, 'prevalence': 0.010782199568712018}
- `[sequence]` gru window=10 sol (n_train_windows=15371): {'tss': 0.6026970057291104, 'hss': 0.1674792824283252, 'pod': 0.6636363636363637, 'far': 0.06093935790725327, 'precision': 0.10610465116279069, 'recall': 0.6636363636363637, 'f1': 0.1829573934837093, 'brier': 0.08088557569733781, 'log_score': 0.33128396709928126, 'pr_auc': 0.14750861441119048, 'roc_auc': 0.8504900371131048, 'tp': 73, 'fp': 615, 'tn': 9477, 'fn': 37, 'n_classes_present_in_y_true': 2, 'prevalence': 0.010782199568712018}

## Outputs

- Manifests: `outputs\real\run_20261004T080014968911Z_seed42\manifests`
- Data: `outputs\real\run_20261004T080014968911Z_seed42\data`
- Labels: `outputs\real\run_20261004T080014968911Z_seed42\labels`
- Features: `outputs\real\run_20261004T080014968911Z_seed42\features`
- Splits: `outputs\real\run_20261004T080014968911Z_seed42\splits`
- Models (checkpoints): `outputs\real\run_20261004T080014968911Z_seed42\models`
- Metrics: `outputs\real\run_20261004T080014968911Z_seed42\metrics`
- Figures: `outputs\real\run_20261004T080014968911Z_seed42\figures`
- Tables: `outputs\real\run_20261004T080014968911Z_seed42\tables`
- Checksums: `outputs\real\run_20261004T080014968911Z_seed42\reports/checksums.json`

[DATA SOURCE: REAL (Martian Chapter 4 mars.run_all pipeline)]