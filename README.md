## Overview

This repository contains the source code developed for the research paper: 
"Techno-Economic Optimization and Hybrid Deep Learning-Based Net Load Forecasting for Intelligent Microgrid Energy Management Using HOMER Pro."
The repository includes the implementation of multiple deep learning and hybrid machine learning models used for net load forecasting, along with the scripts used for data preprocessing, model training, evaluation, and visualization.

## Repository Structure

├── README.md
├── GRU/
├── LSTM+GRU/
├── LSTM+XGBoost/
├── LSTM/
├── TCN+GRU/
├── TCN/

## Models Included

The repository contains implementations of:

- LSTM
- GRU
- LSTM–GRU Hybrid
- LSTM–XGBoost Hybrid
- GRU–TCN Hybrid

Each implementation includes:

- Data preprocessing
- Feature engineering
- Model training
- Performance evaluation
- Forecast visualization
- Net load estimation

## Dataset

The forecasting models require:

- Weather data
- Load demand data

The dataset used in this study was obtained from institutional sources. If redistribution is not permitted, the dataset has not been included in this repository.

Users may use their own datasets while maintaining the same input format and feature names used in the scripts.

## Software Requirements

Python 3.10 or later

Required packages are listed in requirements.txt.

Install them using:

bash
pip install -r requirements.txt

## Running the Code

1. Prepare the weather and load datasets.
2. Update the file paths in the corresponding Python scripts.
3. Execute the desired forecasting model.

Example:

bash
python LSTM_XGBoost.py

Each script reports:

- MAE
- RMSE
- R²
- NMAE
- NRMSE
- MAPE
- SMAPE
- Forecasting latency
- Model size
- Prediction plots

## Reproducibility

Random seeds and preprocessing steps follow the methodology described in the associated publication. Users may modify the train–test split, sequence length, or model hyperparameters for further experimentation.

## Citation

If you use this repository in your research, please cite the associated publication.

## License

This repository is released under the MIT License.

## Contact

For questions regarding the implementation, please contact the corresponding author through the details provided in the published article.
