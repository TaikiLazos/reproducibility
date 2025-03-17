# Benchmarking Jargon Detection on Medical Abstracts

## Overview
This project focuses on benchmarking jargon detection in medical abstracts using transformer-based models. We evaluate different approaches for identifying medical and technical jargon that may be challenging for general readers to understand. Our work builds upon and extends the jargon detection component from the MedREADME paper.

## Setting up the environment

You can set up the environment using the following commands:

```bash
virtualenv medplaba
source medplaba/bin/activate
pip install -r requirements.txt
```

## Experiments

### MedREADME Jargon Detection
We replicate and extend the jargon detection component from the MedREADME paper, focusing on identifying medical terminology that requires simplification. Our implementation supports multiple transformer architectures and provides detailed error analysis.

To run the experiments:
```bash
MODEL="roberta"  # Options: bert, roberta, biobert, pubmedbert

# Run with Python
python src/medreadme.py --model_name $MODEL 

# Or submit as a job
sbatch src/job_files/medreadme.job
```

### PLABA 2024 Task 1a
Implementation of the PLABA 2024 Task 1a for jargon detection:

```bash
python src/plaba.py --experiment_type 1a
```

### Transfer Learning Experiments
Evaluate transfer learning effectiveness between tasks:

```bash
python src/transfer_learning.py
```

## Analysis Tools

### Data Statistics
Generate dataset statistics and visualizations:
```bash
python src/data_statistics.py
```

### Error Analysis
Analyze model predictions and error patterns:
```bash
python src/error_analysis.py --model_name roberta --cls_type 3-cls
```

### Jargon Visualization
Visualize jargon detection results:
```bash
python src/visualize_jargons.py --model_name roberta --cls_type 3-cls --num_samples 5
```

## Data Statistics
```
python src/data_statistic.py
```









