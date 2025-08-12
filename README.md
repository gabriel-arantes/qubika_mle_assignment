# Production-Ready Loan Approval Prediction Service - V2

## 1. Project Objective and Final Outcome

This project documents the complete process of elevating a basic Machine Learning proof-of-concept into a robust, production-grade prediction service. The initial task was to take a simple training script (`model.py`) developed by a teammate and make it ready for deployment. Given that the model is "crucial for the company's success," the focus was on creating a system that is reliable, scalable, maintainable, and available to serve predictions on demand.

The final system is a containerized REST API that serves a machine learning model for predicting loan approvals. The architecture is designed to be cloud-agnostic and incorporates MLOps best practices to manage the entire model lifecycle, from experimentation and versioning to deployment and serving.

This document serves as a comprehensive guide to the architecture, design decisions, execution workflow, and future potential of the system.

## 2. Architectural Decisions and Justifications

In line with the assignment's encouragement to document assumptions and decisions, this section details the "why" behind the final architecture.

#### 2.1. From Script to Service: The Separation of Concerns

* **Initial State:** The original `model.py` was a monolithic script that handled data loading, preprocessing, training, and prediction in one go.
* **Problem:** This approach is unsuitable for production. Training is a resource-intensive, offline activity, while prediction (inference) must be a lightweight, highly available online service.
* **Decision:** The first and most critical decision was to split the codebase into two distinct components:
    1.  **The Training Pipeline (`scripts/`):** A collection of scripts dedicated solely to training models. Its output is not a prediction, but a versioned model artifact managed by a central registry.
    2.  **The Inference API (`app/`):** A high-performance web server whose only job is to load a pre-trained model and serve predictions through a REST API.
* **Justification:** This separation makes the system more robust, scalable, and maintainable. The API can be scaled independently of the training infrastructure, and new models can be developed without impacting the live prediction service.

#### 2.2. Why Docker? The Cloud-Agnostic Approach

* **Requirement:** The model must "work for any cloud provider."
* **Problem:** Different cloud environments (AWS, GCP, Azure) have different underlying operating systems and pre-installed libraries. A model that works locally might fail in the cloud due to dependency conflicts (the "it works on my machine" problem).
* **Decision:** **Docker** was chosen to containerize the inference API.
* **Justification:** The `Dockerfile` creates a self-contained, portable Linux environment with a specific Python version and explicitly defined dependencies. This Docker image can be run, without modification, on any platform that supports Docker, thus achieving true cloud-agnostic deployment.

#### 2.3. Why MLflow? For Reproducibility and Governance

* **Requirement:** This is "model v1 of a series of versions to be deployed."
* **Problem:** Managing multiple model versions, their corresponding training data, parameters, and performance metrics is a major challenge. Without a structured system, it's impossible to reproduce a specific model or reliably compare results.
* **Decision:** **MLflow** was integrated as the core MLOps platform.
* **Justification:**
    * **Experiment Tracking:** MLflow captures everything about a training run: Git commit hash, parameters, performance metrics, and any generated artifacts (like plots). This provides a complete audit trail.
    * **Model Registry:** MLflow provides a central repository for all trained models. It formalizes the model lifecycle through **Aliases** (e.g., `production`, `champion`, `challenger`), providing a governance layer to control which model is served to users. This directly addresses the requirement of managing a series of versions.

#### 2.4. Why Abstract Factory? For Flexibility and Scalability

* **Requirement:** An optional task was to "come up with a better model pipeline."
* **Problem:** A hardcoded pipeline (e.g., `pipeline = Pipeline([SimpleImputer(), LogisticRegression()])`) makes it difficult to experiment with different algorithms or preprocessing techniques.
* **Decision:** The **Abstract Factory** design pattern was implemented for both preprocessing (`preprocessing_factory.py`) and modeling (`model_factory.py`).
* **Justification:** This pattern decouples the main training orchestrator (`train.py`) from the specific implementations. To test a new model (e.g., `XGBoost`), we only need to add an `XGBoostFactory`; no changes are needed in the core training logic. This makes the system highly extensible and encourages rapid experimentation.

#### 2.5. Why FastAPI? For Performance and Usability

* **Requirement:** The model must be "available to receive and predict data upon request."
* **Problem:** A production API needs to be fast, reliable, and easy for client applications to interact with.
* **Decision:** **FastAPI** was chosen to build the REST API.
* **Justification:** FastAPI offers asynchronous request handling for high performance, automatic request/response validation with Pydantic (reducing bugs), and self-generating interactive documentation (Swagger UI), which makes testing and integration significantly easier.

## 3. Project Structure Explained

The project is organized into logical components, each with a single responsibility.

├── app
│   └── main.py                 # The FastAPI application. Loads the production model and exposes prediction endpoints.
├── data
│   └── dataset.csv             # The raw dataset used for training the model.
├── scripts
│   ├── init.py             # Makes the 'scripts' directory a Python package.
│   ├── train.py                # The main training orchestrator. Parses arguments and uses factories to build and train a pipeline.
│   ├── model_factory.py        # Abstract Factory for creating different ML model algorithms (e.g., LogisticRegression, RandomForest).
│   └── preprocessing_factory.py# Abstract Factory for creating different data preprocessing pipelines (e.g., imputation, scaling).
├── .dockerignore               # Specifies files and directories to exclude from the Docker build context.
├── Dockerfile                  # The blueprint for building the production-ready inference API container.
└── requirements.txt            # A list of all Python dependencies required for the project.

## 4. Complete Execution Workflow

This guide provides a comprehensive, step-by-step workflow to run the entire project locally.

### Phase I: Environment Setup

1.  **Prerequisites:**
    * Python 3.9+ and the `pip` package manager.
    * Docker Desktop (or Docker Engine for Linux).
    * A Python virtual environment (e.g., `venv`, `conda`) is highly recommended to isolate dependencies.

2.  **Install Dependencies:**
    From the project's root directory, install all required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the MLflow Server:**
    The MLflow server is the central component for tracking and models. In a terminal, run the following command. This will create a local `mlruns` directory to store all data.
    ```bash
    # From the project root directory
    mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns
    ```
    **Keep this terminal running.** You can access the MLflow dashboard at `http://127.0.0.1:5000`.

### Phase II: Model Experimentation and Promotion

1.  **Run Training Experiments:**
    Open a **new terminal**. The training script is fully parameterized, allowing you to easily test different approaches.
    ```bash
    # Example 1: Train a Random Forest model with simple median imputation
    python -m scripts.train --model-name random_forest --preprocessor-name median_imputer

    # Example 2: Train a Logistic Regression model with imputation and feature scaling
    python -m scripts.train --model-name logistic_regression --preprocessor-name median_imputer_with_scaler
    ```
    Observe the terminal output for performance metrics. Each run will appear in the `loan_approval_v2` experiment in the MLflow UI.

2.  **Promote a Model to Production (Governance Step):**
    This is a deliberate, manual step where a human decides which model is "best" and ready for deployment.
    1.  In the MLflow UI, review the experiment runs. Compare their parameters and metrics (e.g., `accuracy`, `f1_score`).
    2.  Click on the run ID of the model you wish to promote.
    3.  In the run details page, scroll to the **"Artifacts"** section, click the **"model"** folder, and then click the **"Register Model"** button.
    4.  If this is the first time, select "Create New Model" and name it `loan_approval_model`. Otherwise, add it as a new version to the existing model.
    5.  Navigate to the **"Models"** tab from the main sidebar.
    6.  Click on `loan_approval_model`.
    7.  On the desired version, find the **"Aliases"** section. Click to add an alias and type `production`. Press Enter and Save. This tag now officially marks this model version as the one to be used by production services.

### Phase III: Deploying and Testing the API

1.  **Build the Production Docker Image:**
    This command packages the FastAPI application and its dependencies into a self-contained image.
    ```bash
    docker build -t loan-approval-api:latest .
    ```

2.  **Run the Docker Container:**
    We will run the container and use an environment variable (`-e`) to tell the API how to connect to the MLflow server on the host machine.
    ```bash
    # Stop and remove any old container to avoid name conflicts
    docker stop loan-api-container && docker rm loan-api-container

    # Run the new container, linking it to the host's MLflow server
    docker run --name loan-api-container -p 8000:8000 -e MLFLOW_TRACKING_URI="[http://host.docker.internal:5000](http://host.docker.internal:5000)" loan-approval-api:latest
    ```
    * **Why `-e`?** The environment variable `MLFLOW_TRACKING_URI` overrides the default `localhost` address in the code, allowing the container to find the MLflow server via Docker's internal network DNS (`host.docker.internal`).

3.  **Verify and Test the Live API:**
    * **Health Check:** Open `http://localhost:8000/` in your browser. You should see a JSON response confirming that the model with the alias `production` was successfully loaded.
    * **Prediction Test:** Use `curl` or the interactive documentation at `http://localhost:8000/docs` to send a prediction request.
        ```bash
        curl -X 'POST' \
          'http://localhost:8000/api/v2/predict' \
          -H 'Content-Type: application/json' \
          -d '{
            "Age": 45,
            "Annual_Income": 85000,
            "Credit_Score": 650,
            "Loan_Amount": 25000,
            "Loan_Duration_Years": 10,
            "Number_of_Open_Accounts": 5,
            "Had_Past_Default": 0
          }'
        ```
    * **Successful Response:** `{"prediction":"Approved","prediction_code":1}`

## 5. API Documentation

### Health Check Endpoint
* **Path:** `/`
* **Method:** `GET`
* **Description:** Verifies that the API server is running and reports which model (if any) was successfully loaded from the MLflow Model Registry.
* **Success Response (200 OK):**
    ```json
    {
      "status": "online",
      "model_loaded": true,
      "model_name": "loan_approval_model",
      "model_alias": "production"
    }
    ```

### Prediction Endpoint
* **Path:** `/api/v2/predict`
* **Method:** `POST`
* **Description:** Receives loan applicant data in a JSON body and returns a loan approval prediction.
* **Request Body:**
    ```json
    {
      "Age": float,
      "Annual_Income": float,
      "Credit_Score": float,
      "Loan_Amount": float,
      "Loan_Duration_Years": int,
      "Number_of_Open_Accounts": float,
      "Had_Past_Default": int
    }
    ```
* **Success Response (200 OK):**
    ```json
    {
      "prediction": "Approved" | "Não Aprovado",
      "prediction_code": 1 | 0
    }
    ```
* **Error Response (if model is not loaded):**
    ```json
    {
      "error": "Modelo não está disponível. Verifique os logs e se um modelo possui o alias 'production'."
    }
    ```

## 6. Future Improvements & Advanced Integrations

This project provides a robust foundation. The following steps would further enhance its production-readiness.

* **CI/CD Pipeline:** Automate the testing and Docker image build/push process using tools like GitHub Actions or Jenkins. For example, a new push to a `develop` branch could trigger tests, while a merge to `main` could build and push the production image to a container registry (like Docker Hub or AWS ECR).

* **Data and Model Validation Notebook:** Create a `validation.ipynb` Jupyter Notebook. Its purpose would be to:
    * Perform deep exploratory data analysis (EDA) on `dataset.csv`.
    * Use libraries like `pandas-profiling` or `sweetviz` to automatically generate data quality reports.
    * Perform error analysis on model predictions (e.g., where does the model fail most?).
    * This notebook could be run as part of a CI/CD pipeline to detect breaking changes in new data.

* **Enhanced Metrics and Artifact Logging:** Improve the training script to log richer information to MLflow:
    * **Confusion Matrix:** Generate a confusion matrix plot using `matplotlib`/`seaborn` and log it as an image artifact with `mlflow.log_figure()`. This gives immediate visual insight into model performance beyond simple accuracy.
    * **Feature Importance:** For models like Random Forest, calculate and plot feature importances, logging the resulting figure to MLflow. This helps explain model behavior.

* **Dataset Versioning and Tracking:** The current setup does not version the input dataset, `dataset.csv`.
    * **Problem:** If the data changes, model performance will change, and we need to track which model was trained on which version of the data.
    * **Solution:** Integrate a data versioning tool like **DVC (Data Version Control)**. DVC works alongside Git to version large files. We could then log the DVC hash of the dataset as a parameter in our MLflow run, creating an unbreakable link between the model version and the exact data version used to train it.

* **Cloud Deployment:**
    * Deploy the MLflow server to a persistent cloud service (e.g., on an AWS EC2 instance with a managed database like RDS for the backend and S3 for artifacts).
    * Deploy the `loan-approval-api` container to a scalable hosting service like AWS ECS, Google Cloud Run, or Azure Container Apps.