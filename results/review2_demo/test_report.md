# Test Report

**AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)**
Generated 2026-10-07 by `python -m src.run_test_report` - do not edit by hand, regenerate it.

**186 of 186 tests pass** (0 failed, 0 skipped) in 61.0 s.

The whole suite runs on CPU against the synthetic phantom in `src/phantom.py`, so it
needs neither the 12 GB MSD download nor a GPU. That is the point: every stage stays
verifiable on a laptop while the lab RTX 4060 is unavailable.

| Module | Stage | Covers | Tests | Passed | Failed | Time (s) |
|---|---|---|---|---|---|---|
| `test_app.py` | 6.6 | the demo UI's logic, without launching a server. | 15 | 15 | 0 | 10.5 |
| `test_classifier.py` | 6.3 | Stage 6.3 classifier, ROI loading, training loop and the 6.3 -> 6.4 CSV contract. | 18 | 18 | 0 | 19.4 |
| `test_conformal.py` | 6.4 | split conformal prediction. | 11 | 11 | 0 | 0.1 |
| `test_data.py` | 6.1 | dataset discovery and patient-wise splitting. | 7 | 7 | 0 | 0.0 |
| `test_explain.py` | 6.5 | Grad-CAM++. | 12 | 12 | 0 | 1.0 |
| `test_fidelity.py` | 6.5 | the F-Fidelity masking test. | 12 | 12 | 0 | 0.5 |
| `test_metrics.py` | all | Metrics: Dice, the sub-2 cm size bucket, and the stage 6.3/6.4 classification | 17 | 17 | 0 | 0.1 |
| `test_pants_meta.py` | EDA | PanTS structured-report parsing. | 13 | 13 | 0 | 0.1 |
| `test_roi.py` | 6.3 | Stage 6.3 ROI crop. Pure numpy: no MONAI, no data on disk. | 12 | 12 | 0 | 0.0 |
| `test_run_demo_pipeline.py` | 6.1-6.4 | The 6.1 -> 6.5 demo pipeline, end to end on phantoms. | 10 | 10 | 0 | 13.2 |
| `test_run_stage64.py` | 6.4 | Stage 6.4 runner, driven by a synthetic cls_predictions.csv in the agreed | 6 | 6 | 0 | 1.3 |
| `test_selective.py` | 6.6 | the abstention rule. | 12 | 12 | 0 | 0.0 |
| `test_train_seg.py` | 6.2 | model, checkpointing and the training loop. | 16 | 16 | 0 | 13.2 |
| `test_transforms.py` | 6.1 | the preprocessing pipeline itself, on the phantom volumes. | 12 | 12 | 0 | 1.3 |
| `test_uncertainty.py` | 6.4 | MC Dropout. | 13 | 13 | 0 | 0.1 |

## What the suite does not cover

Stated so the table is not read as more than it is:

- **No real-data result.** Every number here comes from the phantom. Dice and AUROC
  measured on it are pipeline demonstrations, not results.
- **No convergence claim.** `test_train_seg.py` checks that the loop, the metric and
  the crash-resume path behave, not that the model reaches the Dice >= 0.75 target.
- **No memory check.** Phantoms are 64x64x40; the 8 GB envelope at 96^3 patches is
  exercised only on the lab PC.
- **No calibration claim.** Synthetic logits look well calibrated because they were
  generated; real ECE is measured on real predictions.

## Per-test detail

| Module | Test | Status | Time (s) |
|---|---|---|---|
| `test_app` | `test_analyse_returns_a_verdict_on_untrained_weights` | passed | 1.48 |
| `test_app` | `test_conformal_set_widens_as_q_hat_grows` | passed | 0.852 |
| `test_app` | `test_explainer_adds_the_fidelity_signal_to_the_verdict` | passed | 6.018 |
| `test_app` | `test_find_checkpoint_returns_none_when_there_is_nothing` | passed | 0.005 |
| `test_app` | `test_list_cases_skips_the_macos_junk` | passed | 0.226 |
| `test_app` | `test_load_volume_applies_stage61_and_finds_the_label` | passed | 0.109 |
| `test_app` | `test_missing_q_hat_drops_the_signal_rather_than_raising` | passed | 0.315 |
| `test_app` | `test_predict_restores_the_models_dropout_rates` | passed | 1.108 |
| `test_app` | `test_q_hat_for_picks_the_nearest_calibrated_alpha` | passed | 0.004 |
| `test_app` | `test_report_always_declares_its_provenance` | passed | 0.003 |
| `test_app` | `test_report_says_when_a_checkpoint_was_loaded` | passed | 0.002 |
| `test_app` | `test_report_shows_the_fidelity_verdict` | passed | 0.002 |
| `test_app` | `test_training_patch_reads_the_checkpoints_own_window` | passed | 0.01 |
| `test_app` | `test_tumor_probability_is_bounded_and_responds_to_the_tumor_channel` | passed | 0.021 |
| `test_app` | `test_without_an_explainer_the_fidelity_signal_stays_absent` | passed | 0.31 |
| `test_classifier` | `test_case_without_pancreas_is_skipped_not_cropped` | passed | 0.094 |
| `test_classifier` | `test_class_weights_balance_and_single_class_warning` | passed | 0.003 |
| `test_classifier` | `test_dropout_reaches_the_network_for_mc_dropout[densenet121]` | passed | 0.285 |
| `test_classifier` | `test_dropout_reaches_the_network_for_mc_dropout[resnet10]` | passed | 0.291 |
| `test_classifier` | `test_evaluate_gives_nan_auroc_for_one_class` | passed | 0.317 |
| `test_classifier` | `test_every_architecture_maps_a_roi_to_two_logits[densenet121]` | passed | 0.465 |
| `test_classifier` | `test_every_architecture_maps_a_roi_to_two_logits[resnet10]` | passed | 0.419 |
| `test_classifier` | `test_every_architecture_maps_a_roi_to_two_logits[resnet18]` | passed | 0.861 |
| `test_classifier` | `test_every_architecture_maps_a_roi_to_two_logits[resnet34]` | passed | 1.544 |
| `test_classifier` | `test_load_roi_cases_labels_and_shapes` | passed | 0.665 |
| `test_classifier` | `test_loss_breaks_ties_between_equal_auroc` | passed | 0.002 |
| `test_classifier` | `test_mc_flag_without_dropout_checkpoint_fails_loudly` | passed | 3.169 |
| `test_classifier` | `test_selection_prefers_auroc_over_loss` | passed | 0.002 |
| `test_classifier` | `test_train_then_predict_writes_the_stage64_contract` | passed | 10.972 |
| `test_classifier` | `test_tumor_probability_is_a_probability_per_case` | passed | 0.321 |
| `test_classifier` | `test_undefined_auroc_ranks_below_any_real_one` | passed | 0.003 |
| `test_classifier` | `test_unknown_architecture_is_rejected` | passed | 0.004 |
| `test_classifier` | `test_without_a_validation_group_it_falls_back_to_training_loss` | passed | 0.004 |
| `test_conformal` | `test_calibrate_matches_the_finite_sample_quantile_by_hand` | passed | 0.003 |
| `test_conformal` | `test_calibrate_rejects_bad_alpha_and_handles_empty` | passed | 0.004 |
| `test_conformal` | `test_coverage_by_hand` | passed | 0.003 |
| `test_conformal` | `test_coverage_curve_rows_are_well_formed` | passed | 0.004 |
| `test_conformal` | `test_marginal_coverage_holds_on_exchangeable_data[0.05]` | passed | 0.024 |
| `test_conformal` | `test_marginal_coverage_holds_on_exchangeable_data[0.1]` | passed | 0.021 |
| `test_conformal` | `test_marginal_coverage_holds_on_exchangeable_data[0.2]` | passed | 0.024 |
| `test_conformal` | `test_nonconformity_accepts_matrix_and_vector` | passed | 0.003 |
| `test_conformal` | `test_predict_set_shapes_and_edge_thresholds` | passed | 0.003 |
| `test_conformal` | `test_qhat_is_monotone_in_alpha` | passed | 0.004 |
| `test_conformal` | `test_summarize_sets_reports_abstention` | passed | 0.003 |
| `test_data` | `test_assert_disjoint_catches_leakage` | passed | 0.004 |
| `test_data` | `test_datalist_is_sorted_and_absolute` | passed | 0.004 |
| `test_data` | `test_datalist_skips_macos_resource_forks` | passed | 0.004 |
| `test_data` | `test_split_is_deterministic` | passed | 0.004 |
| `test_data` | `test_split_is_patient_disjoint_and_complete` | passed | 0.004 |
| `test_data` | `test_split_rejects_bad_fractions` | passed | 0.004 |
| `test_data` | `test_split_roundtrips_through_json` | passed | 0.012 |
| `test_explain` | `test_cam_can_be_asked_for_a_specific_class` | passed | 0.01 |
| `test_explain` | `test_cam_has_the_input_shape_and_is_normalised` | passed | 0.011 |
| `test_explain` | `test_cam_lands_on_the_evidence` | passed | 0.015 |
| `test_explain` | `test_find_target_layer_picks_the_last_convolution` | passed | 0.003 |
| `test_explain` | `test_find_target_layer_refuses_a_model_without_convolutions` | passed | 0.004 |
| `test_explain` | `test_gradcam_works_end_to_end_through_score_head` | passed | 0.151 |
| `test_explain` | `test_hooks_are_removed_on_exit` | passed | 0.007 |
| `test_explain` | `test_model_is_left_in_eval_and_without_stale_gradients` | passed | 0.041 |
| `test_explain` | `test_overlay_is_stamped_and_saved` | passed | 0.584 |
| `test_explain` | `test_score_head_pads_awkward_volume_shapes` | passed | 0.071 |
| `test_explain` | `test_score_head_reduces_a_segmentation_model_to_two_logits` | passed | 0.027 |
| `test_explain` | `test_segmentation_logits_are_rejected_with_a_useful_message` | passed | 0.068 |
| `test_fidelity` | `test_a_faithful_map_passes` | passed | 0.049 |
| `test_fidelity` | `test_a_meaningless_map_fails` | passed | 0.026 |
| `test_fidelity` | `test_figure_is_written` | passed | 0.321 |
| `test_fidelity` | `test_is_deterministic_for_a_fixed_seed` | passed | 0.016 |
| `test_fidelity` | `test_mask_random_blanks_the_requested_fraction` | passed | 0.046 |
| `test_fidelity` | `test_mask_top_k_blanks_the_most_salient_voxels` | passed | 0.003 |
| `test_fidelity` | `test_mask_top_k_rounds_up_so_a_tiny_fraction_still_masks_something` | passed | 0.003 |
| `test_fidelity` | `test_masking_does_not_touch_the_original` | passed | 0.009 |
| `test_fidelity` | `test_result_reports_the_whole_curve` | passed | 0.015 |
| `test_fidelity` | `test_uniform_saliency_fails` | passed | 0.014 |
| `test_fidelity` | `test_verdict_uses_the_aggregate_not_a_vote_per_k` | passed | 0.022 |
| `test_fidelity` | `test_zero_margin_cannot_pass_on_zero_variance` | passed | 0.009 |
| `test_metrics` | `test_auroc_by_size_bucket_ranks_small_tumors_against_all_negatives` | passed | 0.019 |
| `test_metrics` | `test_auroc_perfect_reversed_and_undefined` | passed | 0.009 |
| `test_metrics` | `test_by_size_bucket_reports_overall_and_strata` | passed | 0.014 |
| `test_metrics` | `test_confusion_layout_is_tn_fp_fn_tp` | passed | 0.004 |
| `test_metrics` | `test_dice_bounds` | passed | 0.002 |
| `test_metrics` | `test_dice_half_overlap` | passed | 0.002 |
| `test_metrics` | `test_ece_equals_the_gap_for_one_overconfident_bin` | passed | 0.004 |
| `test_metrics` | `test_ece_is_zero_when_every_bin_is_calibrated` | passed | 0.002 |
| `test_metrics` | `test_ece_weights_bins_by_count` | passed | 0.005 |
| `test_metrics` | `test_equivalent_diameter_recovers_a_known_sphere` | passed | 0.003 |
| `test_metrics` | `test_reliability_curve_keeps_p_equal_to_one_in_the_last_bin` | passed | 0.002 |
| `test_metrics` | `test_reliability_curve_shapes_and_empty_bins_are_nan` | passed | 0.003 |
| `test_metrics` | `test_sensitivity_specificity` | passed | 0.003 |
| `test_metrics` | `test_size_bucket_edges` | passed | 0.002 |
| `test_metrics` | `test_tumor_stats_flags_sub2cm` | passed | 0.003 |
| `test_metrics` | `test_tumor_stats_on_a_tumor_free_case` | passed | 0.002 |
| `test_metrics` | `test_voxel_volume_ml` | passed | 0.002 |
| `test_pants_meta` | `test_clean_report_expands_excel_newlines` | passed | 0.004 |
| `test_pants_meta` | `test_coverage_denominator_is_tumor_cases_not_lesion_cases` | passed | 0.009 |
| `test_pants_meta` | `test_kidney_volume_after_the_lesion_does_not_leak_in` | passed | 0.002 |
| `test_pants_meta` | `test_lesion_table_keeps_the_tumor_flag` | passed | 0.01 |
| `test_pants_meta` | `test_millimetre_sizes_are_converted_to_centimetres` | passed | 0.002 |
| `test_pants_meta` | `test_organ_volume_is_not_mistaken_for_lesion_volume` | passed | 0.002 |
| `test_pants_meta` | `test_parses_a_single_lesion_completely` | passed | 0.002 |
| `test_pants_meta` | `test_parses_multiple_lesions_separately` | passed | 0.003 |
| `test_pants_meta` | `test_report_without_a_lesion_block_yields_nothing` | passed | 0.002 |
| `test_pants_meta` | `test_sub2cm_summary_defaults_to_tumor_cases_only` | passed | 0.015 |
| `test_pants_meta` | `test_sub2cm_summary_reports_the_hardest_stratum` | passed | 0.01 |
| `test_pants_meta` | `test_three_dimensional_size_uses_the_longest_axis` | passed | 0.002 |
| `test_pants_meta` | `test_tumor_and_benign_lesions_are_separated` | passed | 0.008 |
| `test_roi` | `test_accepts_a_channel_first_mask` | passed | 0.002 |
| `test_roi` | `test_bbox_is_clipped_to_the_volume` | passed | 0.002 |
| `test_roi` | `test_bbox_is_tight_without_margin` | passed | 0.002 |
| `test_roi` | `test_channel_axis_is_carried_through` | passed | 0.004 |
| `test_roi` | `test_crop_has_the_requested_shape_and_keeps_the_organ` | passed | 0.004 |
| `test_roi` | `test_crop_values_come_from_the_source_without_resampling` | passed | 0.003 |
| `test_roi` | `test_empty_mask_raises` | passed | 0.003 |
| `test_roi` | `test_margin_is_in_mm_not_voxels` | passed | 0.002 |
| `test_roi` | `test_oversized_organ_is_truncated_not_resized` | passed | 0.002 |
| `test_roi` | `test_shape_mismatch_is_rejected` | passed | 0.002 |
| `test_roi` | `test_tumor_voxels_count_as_pancreas` | passed | 0.002 |
| `test_roi` | `test_window_off_the_volume_edge_is_zero_padded` | passed | 0.003 |
| `test_run_demo_pipeline` | `test_both_size_buckets_are_populated` | passed | 0.007 |
| `test_run_demo_pipeline` | `test_every_stage_is_recorded` | passed | 10.972 |
| `test_run_demo_pipeline` | `test_manifest_admits_the_placeholder` | passed | 0.002 |
| `test_run_demo_pipeline` | `test_missing_predictions_file_is_fatal` | passed | 2.184 |
| `test_run_demo_pipeline` | `test_placeholder_is_deterministic_and_separable` | passed | 0.021 |
| `test_run_demo_pipeline` | `test_placeholder_says_so_in_its_summary` | passed | 0.008 |
| `test_run_demo_pipeline` | `test_predictions_honour_the_63_to_64_contract` | passed | 0.005 |
| `test_run_demo_pipeline` | `test_stage61_artifacts_exist` | passed | 0.004 |
| `test_run_demo_pipeline` | `test_stage61_marks_its_figures_as_phantom` | passed | 0.033 |
| `test_run_demo_pipeline` | `test_stage64_artifacts_exist` | passed | 0.005 |
| `test_run_stage64` | `test_calibration_metrics_are_bounded` | passed | 0.007 |
| `test_run_stage64` | `test_conformal_block_is_consistent` | passed | 0.024 |
| `test_run_stage64` | `test_load_predictions_rejects_missing_columns` | passed | 0.013 |
| `test_run_stage64` | `test_runner_fails_loudly_without_a_calibration_group` | passed | 0.019 |
| `test_run_stage64` | `test_runner_writes_every_deliverable` | passed | 1.231 |
| `test_run_stage64` | `test_summary_reports_overall_and_sub2cm` | passed | 0.002 |
| `test_selective` | `test_confident_positive_and_negative` | passed | 0.002 |
| `test_selective` | `test_decide_frame_adds_columns_and_parses_csv_sets` | passed | 0.01 |
| `test_selective` | `test_degrades_to_a_plain_threshold_with_no_signals` | passed | 0.002 |
| `test_selective` | `test_empty_set_escalates` | passed | 0.002 |
| `test_selective` | `test_failed_fidelity_escalates_but_absent_fidelity_does_not` | passed | 0.002 |
| `test_selective` | `test_labels_phrase_abstention_as_an_action` | passed | 0.003 |
| `test_selective` | `test_mc_spread_escalates` | passed | 0.002 |
| `test_selective` | `test_reasons_accumulate` | passed | 0.002 |
| `test_selective` | `test_summarize_handles_an_empty_table` | passed | 0.003 |
| `test_selective` | `test_summarize_scores_only_the_answered_cases` | passed | 0.01 |
| `test_selective` | `test_threshold_is_inclusive_at_the_boundary` | passed | 0.002 |
| `test_selective` | `test_two_label_set_escalates_however_confident_the_probability_looks` | passed | 0.002 |
| `test_train_seg` | `test_a_dropout_checkpoint_loads_into_a_dropout_free_model` | passed | 0.03 |
| `test_train_seg` | `test_checkpoint_records_the_patch_size_it_was_trained_at` | passed | 0.107 |
| `test_train_seg` | `test_checkpoint_roundtrips_model_and_optimizer` | passed | 0.23 |
| `test_train_seg` | `test_checkpoint_write_is_atomic` | passed | 0.045 |
| `test_train_seg` | `test_dropout_probability_follows_the_argument` | passed | 0.031 |
| `test_train_seg` | `test_metrics_csv_has_one_row_per_epoch_with_both_classes` | passed | 0.021 |
| `test_train_seg` | `test_model_outputs_three_classes_at_input_resolution` | passed | 0.106 |
| `test_train_seg` | `test_model_rejects_mismatched_channels_and_strides` | passed | 0.004 |
| `test_train_seg` | `test_parameter_count_is_positive` | passed | 0.017 |
| `test_train_seg` | `test_parse_patch[64,64,32-expected1]` | passed | 0.006 |
| `test_train_seg` | `test_parse_patch[96-expected0]` | passed | 0.003 |
| `test_train_seg` | `test_resume_continues_after_the_saved_epoch` | passed | 0.067 |
| `test_train_seg` | `test_resume_from_nothing_starts_at_zero` | passed | 0.021 |
| `test_train_seg` | `test_resume_retargets_the_cosine_schedule` | passed | 5.741 |
| `test_train_seg` | `test_resumed_run_extends_history_instead_of_restarting` | passed | 4.001 |
| `test_train_seg` | `test_training_run_produces_all_artifacts` | passed | 2.81 |
| `test_transforms` | `test_augmentation_perturbs_image_and_label_together` | passed | 0.254 |
| `test_transforms` | `test_foreground_crop_removes_air` | passed | 0.044 |
| `test_transforms` | `test_hu_window_maps_the_right_endpoints` | passed | 0.016 |
| `test_transforms` | `test_intensities_are_windowed_to_unit_range` | passed | 0.003 |
| `test_transforms` | `test_labels_survive_resampling_unchanged` | passed | 0.025 |
| `test_transforms` | `test_output_is_channel_first_and_paired` | passed | 0.057 |
| `test_transforms` | `test_patch_sampling_pads_volumes_smaller_than_the_patch` | passed | 0.02 |
| `test_transforms` | `test_patch_sampling_returns_fixed_size_patches` | passed | 0.026 |
| `test_transforms` | `test_reoriented_to_ras` | passed | 0.004 |
| `test_transforms` | `test_resampled_to_target_spacing` | passed | 0.005 |
| `test_transforms` | `test_train_pipeline_with_patches_yields_uniform_cubes` | passed | 0.814 |
| `test_transforms` | `test_validation_pipeline_has_no_random_transforms` | passed | 0.014 |
| `test_uncertainty` | `test_attention_unet_dropout_can_be_switched_on_after_training` | passed | 0.02 |
| `test_uncertainty` | `test_enable_dropout_reports_the_modules_and_leaves_batchnorm_in_eval` | passed | 0.003 |
| `test_uncertainty` | `test_enable_dropout_sets_p_explicitly` | passed | 0.003 |
| `test_uncertainty` | `test_entropy_uses_the_class_axis_not_the_last_one` | passed | 0.005 |
| `test_uncertainty` | `test_explicit_enable_dropout_still_sets_p_permanently` | passed | 0.003 |
| `test_uncertainty` | `test_mc_predict_is_deterministic_when_p_is_zero` | passed | 0.008 |
| `test_uncertainty` | `test_mc_predict_on_a_segmentation_shape_gives_per_voxel_entropy` | passed | 0.009 |
| `test_uncertainty` | `test_mc_predict_rejects_zero_passes` | passed | 0.007 |
| `test_uncertainty` | `test_mc_predict_restores_eval_mode` | passed | 0.009 |
| `test_uncertainty` | `test_mc_predict_restores_the_dropout_rates_it_overrode` | passed | 0.007 |
| `test_uncertainty` | `test_mc_predict_returns_valid_probabilities_with_spread` | passed | 0.05 |
| `test_uncertainty` | `test_mc_table_matches_the_cls_predictions_contract` | passed | 0.015 |
| `test_uncertainty` | `test_predictive_entropy_is_zero_for_certainty_and_max_for_uniform` | passed | 0.004 |
