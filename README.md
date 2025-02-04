# Reproducing the results of MedREADME and PLABA 2024 Task 1

## Setting up the environment

You can set up the environment using the following command:

```
virtualenv medplaba
source medplaba/bin/activate
pip install -r requirements.txt
```

## MedREADME
This project focuses on reproducing the jargon detection component of the MedREADME paper's text simplification task. Using their dataset of medical texts and jargons, we aim to identify and analyze medical jargon that may be difficult for general readers to understand.

Table 8 was replicated using the MedREADME dataset and the configuration of the MedREADME paper.

To run the code, run the following command:
```
python run_experiments.py

# or run with a job script
sbatch src/main.job
```

The results are as follows:

| Models | Binary | 3-Cls | 7-Cls |
|--------|--------|--------|--------|
| Large-size Models |  |  |  |
| BERT | 78.5 | 74.1 | 43.9 |
| RoBERTa | 80.2 | 75.9 | 67.9 |
| BioBERT | 78.4 | 72.6 | 64.9 |
| PubMedBERT | 79.0 | 75.2 | 66.5 |
| Base-size Models |  |  |  |
| BERT | 77.0 | 72.5 | 63.3 |
| RoBERTa | 79.7 | 75.2 | 66.6 |
| BioBERT | 77.1 | 72.8 | 64.1 |
| PubMedBERT | 78.5 | 74.8 | 66.3 |

## PLABA 2024 Task 1

### 1a
Run the following command to train the model:
```
python src/run_plaba_1a.py
```

### 1b
Run the following command to train the model:
```
python src/run_plaba_1b.py
```









