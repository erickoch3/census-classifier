# Salary Classifier API

## Overview

This project is part of the Udacity Machine Learning DevOps Engineer Nanodegree. The goal of this section is to implement a salary classifier based on census data. The application is built using FastAPI, a modern, fast (high-performance) web framework for building APIs with Python 3.6+.

The classifier predicts whether an individual's annual income exceeds $50K based on various demographic features from the census data. The application includes endpoints for performing predictions using a pre-trained machine learning model.

We automatically deploy the API with Github Actions to Elastic Beanstalk after passing linting and testing.

## Project Structure

```plaintext
.
├── app.log                   # Application log file
├── app.py                    # Main FastAPI application
├── conda_requirements.txt    # Conda environment requirements
├── config.json               # Configuration file
├── dvc_on_heroku_install.txt # DVC on Heroku installation instructions
├── environment.yaml          # Conda environment YAML file
├── main.py                   # Main script to run the application
├── Makefile                  # Makefile for automating tasks
├── model_card_template.md    # Template for model card documentation
├── Procfile                  # Heroku Procfile for deployment
├── README.md                 # Project README file
├── requirements.txt          # Project dependencies
├── screenshots               # Directory for storing screenshots
├── scripts                   # Directory for additional scripts
│   └── run_in_conda.sh       # Script to run commands in conda environment
├── src
│   ├── cleaning              # Directory for data cleaning scripts
│   ├── __init__.py           # Init file for src module
│   ├── data.py               # Data processing utilities
│   ├── diagnostics.py        # Diagnostics and validation utilities
│   ├── file_util.py          # Utility for finding repository root
│   ├── logger.py             # Logging utilities
│   ├── model.py              # Model training, saving, loading, and inference
│   ├── score_model.py        # Model scoring script and utilities
│   ├── train_model.py        # Model training script and utilities
├── tests
│   ├── __init__.py           # Init file for tests module
│   ├── test_app.py           # Unit tests for FastAPI endpoints
│   ├── test_file_util.py     # Unit tests for file utilities
│   ├── test_model.py         # Unit tests for model functions
└── model                     # Directory for storing trained models
```

## Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
```

Alternatively, you can create and set up the conda environment using the Makefile:

```bash
make createconda
```

To activate the environment, run:

```bash
conda activate census-classifier
```

## Running the Application

1. **Start the FastAPI server:**

    ```bash
    uvicorn app:app --reload
    ```

2. **Access the API documentation:**

    Visit `http://127.0.0.1:8000/docs` to see the interactive API documentation provided by Swagger UI.

## API Endpoints

### GET /

- **Description:** Welcome message for the API.
- **Response:**
    ```json
    {
        "message": "Welcome to Eric's FastAPI inference service! Let's predict someone's income."
    }
    ```

### POST /predict

- **Description:** Predicts whether an individual's income exceeds $50K based on their demographic features.
- **Request Body:**

    ```json
    {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    ```

- **Response:**

    ```json
    {
        "prediction": ">50K"
    }
    ```

## Running the Tests

To run the tests, use:

```bash
pytest tests
```

## Makefile Commands

The Makefile includes several useful commands for setting up and managing the project environment:

- **Create conda environment:**

    ```bash
    make createconda
    ```

- **Activate conda environment:**

    ```bash
    conda activate census-classifier
    ```

- **Deactivate conda environment:**

    ```bash
    conda deactivate
    ```

- **Install additional dependencies:**

    ```bash
    make install
    ```

- **Lint the codebase:**

    ```bash
    make lint
    ```

- **Automatically format the codebase:**

    ```bash
    make autolint
    ```

- **Run the tests:**

    ```bash
    make test
    ```

- **Clean the data:**

    ```bash
    make clean
    ```

- **Train the model:**

    ```bash
    make train
    ```

- **Score the model:**

    ```bash
    make score
    ```

- **Sanity check:**

    ```bash
    make sanity
    ```

- **Deploy the model:**

    ```bash
    make deploy
    ```

## Continuous Integration and Deployment

This project uses GitHub Actions for continuous integration. The CI pipeline includes:

- Running linting checks
- Running the test suite
- Building the Docker image

On merges to the `master` branch, the project is automatically deployed to AWS Elastic Beanstalk.

## Author

Eric Koch

## Date Created

2024-05-30

## License

This project is licensed under the MIT License.