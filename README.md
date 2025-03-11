# Reproducing the results of MedREADME and PLABA 2024 Task 1

## Setting up the environment

You can set up the environment using the following command:

```
virtualenv medplaba
source medplaba/bin/activate
pip install -r requirements.txt
```

## MedReadMe
This project focuses on reproducing the jargon detection component of the MedREADME paper's text simplification task. Using their dataset of medical texts and jargons, we aim to identify and analyze medical jargon that may be difficult for general readers to understand.

Table 8 was replicated using the MedREADME dataset and the configuration of the MedREADME paper.

To run the code, run the following command:
```
MODEL="roberta"  # You can choose from bert, roberta, biobert, pubmedbert

# Run the script
python src/medreadme.py  --model_name $MODEL 

# or run with a job script
sbatch src/job_files/medreadme.job
```


## PLABA 2024 Task 1

### 1a
Run the following command to train the model:
```
python src/plaba.py  --experiment_type 1a
```

## Transfer Learning
Run the following command to replicate my results:
```
python src/transfer_learning.py
```

## Data Statistics
```
python src/data_statistic.py
```









