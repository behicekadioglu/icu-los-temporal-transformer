# ICU Length-of-Stay Prediction with Temporal Transformers

Predicting **Intensive Care Unit (ICU) length of stay (LoS)** from longitudinal clinical data using transformer-based deep learning and machine learning baselines.

This project investigates whether a **Temporal Transformer** can effectively model clinical time-series data derived from **MIMIC-IV** for ICU length-of-stay regression. The transformer-based approach is evaluated alongside traditional machine learning and neural network baselines within a structured experimental pipeline.

---

## Overview

ICU length of stay is an important outcome for hospital resource planning, bed management, and clinical operations.

Electronic health records contain large amounts of longitudinal clinical information, including physiological measurements and laboratory observations. Modeling these measurements as temporal sequences provides an opportunity to capture changes in a patient's clinical state over time.

The objective of this project is to:

* preprocess longitudinal ICU data,
* construct fixed-length temporal representations,
* identify informative clinical features,
* train a transformer-based regression model,
* compare it with established machine learning baselines,
* optimize model hyperparameters,
* evaluate performance using cross-validation,
* and investigate the contribution of model components through ablation studies.

---

## Models

The project evaluates four regression approaches.

### Temporal Transformer

The primary model uses a **Transformer-based architecture** to model relationships across clinical time steps.

Self-attention allows the model to learn dependencies between observations occurring at different points in the ICU stay rather than relying exclusively on sequential recurrence.

### XGBoost

**XGBoost** is included as a strong tree-based baseline for structured clinical data.

It is also used during feature-selection experiments to estimate the relative importance of candidate variables.

### Multilayer Perceptron

A feed-forward **Multilayer Perceptron (MLP)** provides a neural-network baseline without an explicit temporal attention mechanism.

### Ridge Regression

**Ridge Regression** provides a regularized linear baseline and helps determine how much improvement more complex nonlinear models provide over a simpler approach.

---

## Dataset

This project uses data derived from **MIMIC-IV**, a large de-identified electronic health record dataset containing information from patients treated at Beth Israel Deaconess Medical Center.

MIMIC-IV contains clinical information such as:

* ICU admission information,
* physiological measurements,
* laboratory observations,
* demographic information,
* and other structured hospital records.

### Data Access

MIMIC-IV is **not distributed with this repository**.

Researchers must obtain access separately through PhysioNet and comply with the corresponding credentialing and data-use requirements.

Because the original clinical data cannot simply be redistributed with the project, generated datasets and local data directories should not be committed to the repository.

---

## Problem Formulation

The task is formulated as a **regression problem**.

For each eligible ICU stay, a sequence of clinical observations is used to estimate the patient's ICU length of stay.

Conceptually:

```text
Clinical time-series
        │
        ▼
Preprocessing
        │
        ▼
Temporal representation
        │
        ├───────────────┐
        ▼               ▼
Temporal Transformer   Baseline Models
        │               │
        └───────┬───────┘
                ▼
      Length-of-Stay Prediction
                │
                ▼
            Evaluation
```

---

## Experimental Pipeline

The overall experimental workflow consists of several stages.

### 1. Data Extraction

Relevant clinical variables are extracted from MIMIC-IV and associated with individual ICU stays.

### 2. Data Cleaning

The preprocessing pipeline handles issues commonly encountered in longitudinal clinical data, including:

* missing observations,
* inconsistent sampling,
* invalid values,
* and differences in feature availability.

### 3. Temporal Representation

Clinical measurements are transformed into representations suitable for temporal modeling.

Each ICU stay is represented using a sequence of clinical observations across time.

### 4. Feature Selection

Feature-selection experiments are performed to reduce the dimensionality of the input and identify variables that contribute most strongly to prediction.

XGBoost feature importance is used as part of this analysis.

### 5. Model Training

The following models are trained and compared:

* Temporal Transformer
* XGBoost
* Multilayer Perceptron
* Ridge Regression

### 6. Hyperparameter Optimization

**Optuna** is used to explore model hyperparameters and identify configurations that perform well on validation data.

### 7. Cross-Validation

Cross-validation is used to estimate model performance across multiple data partitions rather than relying on a single train/test split.

### 8. Ablation Studies

Ablation experiments are used to investigate how individual components of the modeling pipeline affect predictive performance.

---

## Evaluation Metrics

Model performance is evaluated using multiple regression metrics.

### Mean Absolute Error — MAE

MAE measures the average absolute difference between predicted and actual values.

Lower values indicate better predictions.

### Root Mean Squared Error — RMSE

RMSE penalizes larger prediction errors more strongly than MAE.

Lower values indicate better performance.

### Coefficient of Determination — R²

R² measures how much of the variation in ICU length of stay is explained by the model.

Higher values indicate better explanatory performance.

### Median Absolute Error — MedAE

MedAE reports the median absolute prediction error and is less sensitive to large outliers than MAE or RMSE.

---

## Model Comparison

The main experimental comparison includes:

| Model                | Type                           | Temporal Modeling              |
| -------------------- | ------------------------------ | ------------------------------ |
| Ridge Regression     | Linear baseline                | No                             |
| MLP                  | Neural-network baseline        | No explicit temporal attention |
| XGBoost              | Gradient-boosted trees         | No explicit temporal attention |
| Temporal Transformer | Attention-based neural network | Yes                            |

The goal of the comparison is not only to identify the model with the lowest prediction error, but also to determine whether explicit temporal modeling provides measurable benefits over strong tabular baselines.

---

## Results

Experimental outputs are stored in the `results/` directory.

The evaluation focuses on:

* predictive accuracy,
* comparison between the Temporal Transformer and baseline models,
* model stability across validation folds,
* hyperparameter optimization results,
* feature-selection experiments,
* and ablation studies.

The main metrics reported throughout the experiments are:

```text
MAE
RMSE
R²
MedAE
```

A final model comparison table can be added here once the definitive experimental configuration has been selected.

---

## Ablation Studies

Ablation studies are included to investigate which elements of the pipeline contribute to model performance.

These experiments help answer questions such as:

* How much does feature selection affect performance?
* Does the temporal architecture improve prediction compared with non-temporal baselines?
* How sensitive is the model to architectural choices?
* Which components contribute most strongly to the final result?

Ablation analysis is especially useful because overall predictive performance alone does not explain why a model performs well.

---

## Feature Selection

Clinical datasets can contain many correlated or weakly informative variables.

Feature selection is therefore used to investigate whether a smaller set of informative variables can preserve or improve predictive performance.

The pipeline uses **XGBoost feature importance** as part of this analysis.

The selected feature subset is then used in subsequent modeling experiments.

### Methodological Note

Supervised feature-selection procedures that use the prediction target should be fitted only using training data when performing strict out-of-sample evaluation.

For a fully leakage-controlled evaluation pipeline, feature selection, preprocessing parameters, and hyperparameter optimization should all be learned from the appropriate training partitions before evaluation on the corresponding held-out data.

---

## Repository Structure

The repository is organized around preprocessing, training, evaluation, and experimental outputs.

```text
icu-los-temporal-transformer/
│
├── scripts/
│   ├── preprocessing and feature-selection code
│   ├── baseline model training
│   ├── transformer training
│   └── evaluation utilities
│
├── results/
│   ├── model outputs
│   ├── evaluation results
│   └── generated figures
│
├── README.md
├── requirements.txt
└── .gitignore
```

Local MIMIC-IV data is intentionally excluded from version control.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/behicekadioglu/icu-los-temporal-transformer.git
cd icu-los-temporal-transformer
```

Creating a virtual environment is recommended.

### Using `venv`

```bash
python -m venv .venv
```

Activate the environment.

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Main Dependencies

The project uses the following core libraries:

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* PyTorch
* XGBoost
* Optuna
* Matplotlib
* Seaborn

Exact dependency versions should be pinned in `requirements.txt` when creating a fully reproducible release.

---

## Running the Project

The general workflow is:

```text
1. Obtain authorized MIMIC-IV access
2. Prepare the required local data
3. Run preprocessing
4. Generate the model-ready dataset
5. Train baseline models
6. Train the Temporal Transformer
7. Run hyperparameter optimization
8. Evaluate the models
9. Run ablation experiments
10. Inspect the generated results
```

### Preprocessing

The preprocessing pipeline is implemented under the `scripts/` directory.

For example:

```bash
python scripts/preprocess.py
```

The preprocessing stage prepares the data required by the training pipeline.

> The exact local MIMIC-IV paths may need to be configured according to the user's environment.

### Training

Training scripts under `scripts/` are used to train the individual baseline and transformer models.

The available scripts should be run according to the experimental configuration being evaluated.

### Results

Generated metrics, experimental outputs, and figures are written to the `results/` directory.

---

## Reproducibility

The project is designed to make experiments repeatable through:

* explicit preprocessing code,
* shared training pipelines,
* cross-validation,
* automated hyperparameter optimization,
* saved experimental outputs,
* and consistent evaluation metrics.

For stronger reproducibility, the following should also be fixed and documented for final experiments:

* package versions,
* Python version,
* random seeds,
* model hyperparameters,
* preprocessing configuration,
* dataset version,
* hardware environment where relevant,
* and exact train/validation/test procedures.

---

## Data Leakage Considerations

Preventing information leakage is particularly important when evaluating machine learning models on clinical data.

Any operation that learns information from the target or the distribution of the complete dataset should be considered part of the model-fitting process.

A strict evaluation workflow should therefore follow a structure similar to:

```text
Outer Cross-Validation Fold
│
├── Training Partition
│   │
│   ├── Fit preprocessing
│   ├── Perform supervised feature selection
│   ├── Run inner validation / hyperparameter search
│   └── Train final fold model
│
└── Test Partition
    │
    ├── Apply transformations learned from training
    └── Evaluate final predictions
```

The held-out test partition should not influence feature selection, hyperparameter optimization, or preprocessing parameters.

---

## Why Temporal Transformers?

Traditional tabular models typically represent an ICU stay using an aggregated feature vector.

However, clinical measurements are inherently temporal.

For example, two patients may have similar average values while exhibiting very different trajectories over time.

Transformer-based architectures can use self-attention to learn relationships between different positions in a sequence.

This makes them an interesting approach for investigating whether temporal clinical trajectories contain predictive information that is lost when observations are reduced to static summaries.

---

## Limitations

Several limitations should be considered when interpreting the results.

### Single Data Source

The experiments are based on MIMIC-IV.

Performance on data from another hospital or healthcare system may differ because of differences in:

* patient populations,
* clinical practices,
* measurement frequency,
* available variables,
* and documentation systems.

### Missing Clinical Data

Electronic health records contain irregularly sampled and missing observations.

Preprocessing decisions can therefore influence model performance.

### Length-of-Stay Distribution

ICU length of stay can contain highly skewed values and extreme cases.

For this reason, multiple regression metrics are considered rather than relying on a single performance measure.

### Predictive Rather Than Causal Modeling

The models are designed to identify predictive relationships.

They should not be interpreted as establishing causal relationships between clinical variables and ICU length of stay.

---

## Clinical Use Disclaimer

This repository is an **academic and research project**.

The models are not validated medical devices and are **not intended for clinical decision-making or direct patient care**.

Any real-world clinical use would require substantially more validation, including external validation, calibration analysis, bias assessment, prospective evaluation, and appropriate regulatory review.

---

## Future Work

Potential extensions of the project include:

* moving all supervised feature selection fully inside cross-validation folds,
* external validation using an independent ICU dataset,
* comparison with additional temporal architectures,
* improved handling of irregularly sampled observations,
* uncertainty estimation,
* model calibration analysis,
* explainability methods for temporal predictions,
* subgroup performance analysis,
* systematic investigation of clinically meaningful feature groups,
* and more extensive statistical comparison between models.

---

## Technologies

### Machine Learning

`PyTorch` · `Scikit-learn` · `XGBoost` · `Optuna`

### Data Processing

`NumPy` · `Pandas` · `SciPy`

### Visualization

`Matplotlib` · `Seaborn`

### Development

`Python` · `Git` · `GitHub`

---

## Project Goals

This project was developed to explore several topics at the intersection of machine learning and healthcare:

* modeling longitudinal clinical data,
* transformer architectures for time-series regression,
* comparison between deep learning and classical machine learning,
* systematic model evaluation,
* hyperparameter optimization,
* feature selection,
* and experimental ablation.

The broader goal is to better understand both the advantages and limitations of applying modern temporal models to structured ICU data.

---

## Author

**Behice Kadıoğlu**

Computer Engineering
İzmir Institute of Technology

GitHub: [@behicekadioglu](https://github.com/behicekadioglu)

---

## Acknowledgements

This project uses **MIMIC-IV**, made available through PhysioNet.

MIMIC-IV contains de-identified clinical data and requires users to comply with the access and data-use conditions established by the dataset providers.

The project also relies on several open-source scientific Python libraries, including PyTorch, Scikit-learn, XGBoost, and Optuna.

---

## License

No license is currently specified unless a `LICENSE` file is included in this repository.

If the project is intended to be reusable by others, an appropriate open-source license should be selected and added explicitly.

Note that an open-source license for the code does **not** grant redistribution rights for MIMIC-IV data. Dataset access remains subject to the separate MIMIC-IV and PhysioNet requirements.
