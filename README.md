# Workload Migration Optimization with ConvFormerNet and BOHB

## Overview
This project focuses on analyzing workload migration and resource allocation in cloud computing using a deep learning approach. The primary goal is to predict the maximum memory usage during workload migration to optimize resource distribution. A custom CNN-Transformer model, **ConvFormerNet**, is employed, coupled with **Bayesian Optimization Hyperband (BOHB)** for hyperparameter tuning.

## Dataset
The dataset consists of cloud workload traces with various fields, including:

- **time**: Timestamps of events
- **instance_events_type**: Type of workload events
- **scheduling_class**: Scheduling class type
- **priority**: Priority of the instance
- **average_usage** and **maximum_usage**: CPU and memory usage statistics
- **cpu_usage_distribution**: Array of CPU usage data
- **assigned_memory**, **page_cache_memory**, **cycles_per_instruction**, **memory_accesses_per_instruction**: Memory and CPU performance metrics

Additional feature engineering and transformations were applied to extract useful metrics for model training.

## Installation
To set up the environment, ensure that the following Python libraries are installed:

```bash
pip install pandas numpy seaborn matplotlib sklearn torch tensorflow hpbandster ConfigSpace
```

## Data Preprocessing

The data preprocessing steps include:

1. **Selecting Relevant Columns**: Retaining columns relevant to resource usage.
2. **Feature Extraction**: Extracting CPU and memory usage data from nested dictionaries and arrays.
3. **Imputing Missing Values**: Using mean imputation for features like `cycles_per_instruction`.
4. **Scaling**: Normalizing feature values with `MinMaxScaler` for improved model convergence.

## Model Architecture: ConvFormerNet

**ConvFormerNet** combines Convolutional Neural Network (CNN) and Transformer Encoder layers to capture both spatial and sequential patterns in workload data. The architecture includes:

- **Convolutional Layers**: For extracting spatial features
- **Transformer Encoder Block**: For capturing sequential dependencies
- **Global Pooling and Dense Layers**: Final output for regression tasks

## Hyperparameter Optimization

**BOHB (Bayesian Optimization Hyperband)** is used to tune hyperparameters, specifically:

- **conv_filters**: Number of filters in convolutional layers
- **num_heads**: Number of attention heads in the transformer
- **ff_dim**: Dimensionality of the feedforward network
- **dropout_rate**: Dropout rate for regularization

## Training

The dataset is split into training, validation, and testing sets. Key steps in training include:

1. **Feature Scaling**: Normalizing features with `MinMaxScaler`.
2. **Training**: Training the ConvFormerNet model using BOHB for hyperparameter optimization.
3. **Evaluation**: Evaluating the model using Mean Absolute Error (MAE) as the primary metric.

## Results

The model tuning and training resulted in the following best hyperparameters:

- **conv_filters**: 48
- **dropout_rate**: 0.05
- **ff_dim**: 243
- **num_heads**: 6

Performance metrics on the test set:

- **MAE**: 0.000578
- **RMSE**: 0.000973
- **R-Squared**: 0.997

## Visualizations

1. **Histograms**: Visualize data distribution for key features.
   ![Hist](https://github.com/user-attachments/assets/f18e3f8b-853f-48f2-b5c0-e3bf3e3346c6)
2. **Scatter Plots**: Analyze relationships between memory allocation and CPU usage.
   ![Sactter](https://github.com/user-attachments/assets/b52cd59c-1069-4887-9ff2-0ebf66a7af4f)
3. **Correlation Heatmap**: Visualize dependencies among features.
   ![Corr](https://github.com/user-attachments/assets/b1a9615b-3af2-42ff-b98f-5b516e70a156)
4. **Training vs. Validation Loss Plot**: Monitor model performance over epochs.
   ![Model](https://github.com/user-attachments/assets/d3a79529-f03e-413d-8e2a-cc3b7c5cd734)
5. **Actual vs. Predicted Plot**: Compare actual and predicted values for selected samples.
   ![Actual](https://github.com/user-attachments/assets/4289f093-7b06-4a64-9026-579dbd54c0bf)

