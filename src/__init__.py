"""Reusable code for the pancreatic cancer early-detection pipeline.

Modules follow the stage numbering in CLAUDE.md:
    config      constants from Chapter 6 of the report
    data        6.1  dataset discovery, patient-wise splits
    transforms  6.1  preprocessing and augmentation
    metrics     all  Dice, tumor size, sub-2 cm stratification
    viz         all  review-deck figures
"""
__version__ = "0.1.0"
