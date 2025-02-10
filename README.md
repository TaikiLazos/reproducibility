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
sbatch src/job_files/medreadme_entity_level.job 
sbatch src/job_files/medreadme_token_level.job
```

The results are as follows:

Entity Level:

| Models | Binary | 3-Cls | 7-Cls |
|--------|--------|--------|--------|
| Large-size Models |  |  |  |
| BERT | 87.9 | 86.5 | 85.5 |
| RoBERTa | 91.3 | 89.0 | 82.7 |
| BioBERT | ??? | ??? | ??? |
| PubMedBERT | 87.3 | 86.7 | 83.8 |
| Base-size Models |  |  |  |
| BERT | 87.3 | 83.1 | 82.3 |
| RoBERTa | 89.0 | 85.7 | 83.4 |
| BioBERT | ??? | ??? | ??? |
| PubMedBERT | 87.7 | 85.4 | 83.2 |

Token Level:

| Models | Binary | 3-Cls | 7-Cls |
|--------|--------|--------|--------|
| Large-size Models |  |  |  |
| BERT | 88.7 | 87.9 | 79.1 |
| RoBERTa | ??? | ??? | ??? |
| BioBERT | 88.8 | 87.4 | 0.0 |
| PubMedBERT | 87.1 | 85.4 | 81.7 |
| Base-size Models |  |  |  |
| BERT | 89.0 | 87.0 | 78.6 |
| RoBERTa | ??? | ??? | ??? |
| BioBERT | 90.2 | 88.2 | 85.3 |
| PubMedBERT | 86.7 | 84.1 | 79.8 |

## PLABA 2024 Task 1

### 1a
Run the following command to train the model:
```
python src/finetune.py
python src/run_plaba_1a.py
```

Fine-tuned the RoBERTa base model from the medreadme paper's on training set from PLABA 2024 Task 1 dataset and tested on the test set.

```
=== Evaluation Results ===
Total abstracts processed: 300
Precision: 40.71%
Recall: 25.97%
F1 Score: 31.71%
```


### 1b
Run the following command to train the model:
```
python src/run_plaba_1b.py
```









