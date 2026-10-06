# Test Report

**AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)**
Generated 2026-10-06 by `python -m src.run_test_report` - do not edit by hand, regenerate it.

**56 of 56 tests pass** (0 failed, 0 skipped) in 25.6 s.

The whole suite runs on CPU against the synthetic phantom in `src/phantom.py`, so it
needs neither the 12 GB MSD download nor a GPU. That is the point: every stage stays
verifiable on a laptop while the lab RTX 4060 is unavailable.

| Module | Stage | Covers | Tests | Passed | Failed | Time (s) |
|---|---|---|---|---|---|---|
| `test_data.py` | 6.1 | dataset discovery and patient-wise splitting. | 7 | 7 | 0 | 0.5 |
| `test_metrics.py` | all | Metrics: Dice and the sub-2 cm size bucket. | 8 | 8 | 0 | 0.1 |
| `test_pants_meta.py` | EDA | PanTS structured-report parsing. | 13 | 13 | 0 | 0.1 |
| `test_train_seg.py` | 6.2 | model, checkpointing and the training loop. | 16 | 16 | 0 | 23.4 |
| `test_transforms.py` | 6.1 | the preprocessing pipeline itself, on the phantom volumes. | 12 | 12 | 0 | 1.5 |

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
| `test_data` | `test_assert_disjoint_catches_leakage` | passed | 0.006 |
| `test_data` | `test_datalist_is_sorted_and_absolute` | passed | 0.003 |
| `test_data` | `test_datalist_skips_macos_resource_forks` | passed | 0.437 |
| `test_data` | `test_split_is_deterministic` | passed | 0.008 |
| `test_data` | `test_split_is_patient_disjoint_and_complete` | passed | 0.003 |
| `test_data` | `test_split_rejects_bad_fractions` | passed | 0.003 |
| `test_data` | `test_split_roundtrips_through_json` | passed | 0.032 |
| `test_metrics` | `test_by_size_bucket_reports_overall_and_strata` | passed | 0.064 |
| `test_metrics` | `test_dice_bounds` | passed | 0.005 |
| `test_metrics` | `test_dice_half_overlap` | passed | 0.004 |
| `test_metrics` | `test_equivalent_diameter_recovers_a_known_sphere` | passed | 0.004 |
| `test_metrics` | `test_size_bucket_edges` | passed | 0.004 |
| `test_metrics` | `test_tumor_stats_flags_sub2cm` | passed | 0.005 |
| `test_metrics` | `test_tumor_stats_on_a_tumor_free_case` | passed | 0.005 |
| `test_metrics` | `test_voxel_volume_ml` | passed | 0.005 |
| `test_pants_meta` | `test_clean_report_expands_excel_newlines` | passed | 0.004 |
| `test_pants_meta` | `test_coverage_denominator_is_tumor_cases_not_lesion_cases` | passed | 0.011 |
| `test_pants_meta` | `test_kidney_volume_after_the_lesion_does_not_leak_in` | passed | 0.002 |
| `test_pants_meta` | `test_lesion_table_keeps_the_tumor_flag` | passed | 0.009 |
| `test_pants_meta` | `test_millimetre_sizes_are_converted_to_centimetres` | passed | 0.003 |
| `test_pants_meta` | `test_organ_volume_is_not_mistaken_for_lesion_volume` | passed | 0.002 |
| `test_pants_meta` | `test_parses_a_single_lesion_completely` | passed | 0.003 |
| `test_pants_meta` | `test_parses_multiple_lesions_separately` | passed | 0.002 |
| `test_pants_meta` | `test_report_without_a_lesion_block_yields_nothing` | passed | 0.003 |
| `test_pants_meta` | `test_sub2cm_summary_defaults_to_tumor_cases_only` | passed | 0.021 |
| `test_pants_meta` | `test_sub2cm_summary_reports_the_hardest_stratum` | passed | 0.009 |
| `test_pants_meta` | `test_three_dimensional_size_uses_the_longest_axis` | passed | 0.002 |
| `test_pants_meta` | `test_tumor_and_benign_lesions_are_separated` | passed | 0.011 |
| `test_train_seg` | `test_a_dropout_checkpoint_loads_into_a_dropout_free_model` | passed | 0.041 |
| `test_train_seg` | `test_checkpoint_records_the_patch_size_it_was_trained_at` | passed | 0.131 |
| `test_train_seg` | `test_checkpoint_roundtrips_model_and_optimizer` | passed | 5.742 |
| `test_train_seg` | `test_checkpoint_write_is_atomic` | passed | 0.057 |
| `test_train_seg` | `test_dropout_probability_follows_the_argument` | passed | 0.043 |
| `test_train_seg` | `test_metrics_csv_has_one_row_per_epoch_with_both_classes` | passed | 0.021 |
| `test_train_seg` | `test_model_outputs_three_classes_at_input_resolution` | passed | 0.314 |
| `test_train_seg` | `test_model_rejects_mismatched_channels_and_strides` | passed | 0.003 |
| `test_train_seg` | `test_parameter_count_is_positive` | passed | 0.018 |
| `test_train_seg` | `test_parse_patch[64,64,32-expected1]` | passed | 0.003 |
| `test_train_seg` | `test_parse_patch[96-expected0]` | passed | 0.003 |
| `test_train_seg` | `test_resume_continues_after_the_saved_epoch` | passed | 0.127 |
| `test_train_seg` | `test_resume_from_nothing_starts_at_zero` | passed | 0.021 |
| `test_train_seg` | `test_resume_retargets_the_cosine_schedule` | passed | 7.566 |
| `test_train_seg` | `test_resumed_run_extends_history_instead_of_restarting` | passed | 4.701 |
| `test_train_seg` | `test_training_run_produces_all_artifacts` | passed | 4.653 |
| `test_transforms` | `test_augmentation_perturbs_image_and_label_together` | passed | 0.182 |
| `test_transforms` | `test_foreground_crop_removes_air` | passed | 0.045 |
| `test_transforms` | `test_hu_window_maps_the_right_endpoints` | passed | 0.004 |
| `test_transforms` | `test_intensities_are_windowed_to_unit_range` | passed | 0.004 |
| `test_transforms` | `test_labels_survive_resampling_unchanged` | passed | 0.006 |
| `test_transforms` | `test_output_is_channel_first_and_paired` | passed | 0.062 |
| `test_transforms` | `test_patch_sampling_pads_volumes_smaller_than_the_patch` | passed | 0.035 |
| `test_transforms` | `test_patch_sampling_returns_fixed_size_patches` | passed | 0.023 |
| `test_transforms` | `test_reoriented_to_ras` | passed | 0.004 |
| `test_transforms` | `test_resampled_to_target_spacing` | passed | 0.003 |
| `test_transforms` | `test_train_pipeline_with_patches_yields_uniform_cubes` | passed | 1.145 |
| `test_transforms` | `test_validation_pipeline_has_no_random_transforms` | passed | 0.013 |
