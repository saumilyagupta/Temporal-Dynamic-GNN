# Experimental Report - 3D Graph Convolutional Network for Video Anomaly Detection

This folder contains all materials related to the experimental report for the 3D-GCN video anomaly detection experiment.

## Contents

### Main Report Files

- **`data_report.tex`** - LaTeX source file for concise data-focused report (ALL DATA ONLY)
- **`data_report.pdf`** - Compiled PDF version of data report (117 KB, 6 pages)
- **`experiment_report.tex`** - LaTeX source file for the comprehensive experimental report
- **`experiment_report.pdf`** - Compiled PDF version of the detailed report (243 KB, 12 pages)

### Supporting Files

- **`analyze_timestamps.py`** - Python script used to analyze timestamp ranges and frame counts across the dataset

## Report Types

### Data Report (`data_report.pdf`)
**Concise, data-focused report with all experimental data:**
- Model configuration tables
- Dataset statistics
- Complete training history (all 21 epochs)
- Test set results and metrics
- Individual predictions
- Confusion matrix
- Training summary statistics
- Layer dimensions and architecture details

**No theoretical explanations - just the data and numbers.**

### Detailed Report (`experiment_report.pdf`)
**Comprehensive academic-style report including:**

The detailed report includes:

1. **Abstract** - Executive summary of the experiment
2. **Introduction** - Problem statement and motivation
3. **Methodology** - Approach and problem formulation
4. **Model Architecture** - Detailed mathematical description of the 3D-GCN
5. **Dataset** - Comprehensive dataset statistics and characteristics
6. **Experimental Setup** - Training configuration and hyperparameters
7. **Results and Analysis** - Performance metrics, training history, and detailed analysis
8. **Discussion** - Strengths, limitations, and insights
9. **Future Improvements** - Suggestions for enhancement
10. **Conclusion** - Summary and contributions
11. **Appendices** - Implementation details and complete configurations

## Key Findings

- **Dataset**: 53 video samples (37 train, 6 validation, 10 test)
- **Frame Range**: 22-79 frames per sample (mean ~50)
- **Timestamp Range**: 0.0-3.12 seconds (normalized per-sample to [0,1])
- **Test Performance**: 50% accuracy, 40% recall, 44.4% F1-score
- **Model Architecture**: 3 GCN layers, 64 hidden dimensions, all pooling (mean+max+sum)

## Compiling the Report

To regenerate the PDF from the LaTeX source:

```bash
cd report
pdflatex experiment_report.tex
pdflatex experiment_report.tex  # Run twice for cross-references
```

## Related Files

The actual experimental results (plots, metrics, predictions) are stored in:
- `../results/predictions/` - Test set predictions, confusion matrix, ROC/PR curves
- `../results/logs/` - Training logs and curves
- `../results/checkpoints/` - Model checkpoints

## Report Statistics

- **Pages**: 12
- **Sections**: 11 main sections + 2 appendices
- **Tables**: 8 detailed tables
- **Mathematical Formulations**: Complete model architecture equations

