# Taxi Time Prediction

## Virtual environment  setup

Run the following commands in the terminal to set up the environment
```bash
pip install virtualenv
python3 -m venv taxi_time
source taxi_time/bin/activate
pip install -r requirements.txt
```

## Preprocessing

Preprocessing was done using three python files:
- ***create_processed_data.py***
- ***preprocess.py***
- ***merging.py***

The ***create_preprocessed_data.py*** reads in all the input data and creates a new ***df_preprocessed_2015-2019.csv*** file that is used in modelling using the functions defined in ***preprocess.py*** and ***merging.py***

To create the final dataset used for modelling run the following command on the terminal
```bash
python create_processed_data.py
```
## Modelling

LightGBM was used to model the data. The model can be trained using the Run_lightGBM.py file which trains the model and prints the scores.
```bash
python Run_lightGBM.py
```
## Usage

For running both preprocessing and training of the model, run the following:

```bash
python run_preprocess_and_model.py
```



