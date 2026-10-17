
import os
import sys
from urllib.parse import urlparse

import dagshub
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)


# DagsHub configuration
DAGSHUB_REPO_OWNER = "Mu3tafa-python"
DAGSHUB_REPO_NAME = "networksecurity"

MLFLOW_TRACKING_URI = (
    f"https://dagshub.com/"
    f"{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
)

# Configure MLflow tracking
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

# Initialize DagsHub
dagshub.init(
    repo_owner=DAGSHUB_REPO_OWNER,
    repo_name=DAGSHUB_REPO_NAME,
    mlflow=True,
)


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def track_mlflow(
        self,
        best_model,
        classification_train_metric,
        classification_test_metric,
    ):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

            with mlflow.start_run():

                # Log training metrics
                mlflow.log_metric(
                    "train_f1_score",
                    classification_train_metric.f1_score,
                )
                mlflow.log_metric(
                    "train_precision",
                    classification_train_metric.precision_score,
                )
                mlflow.log_metric(
                    "train_recall",
                    classification_train_metric.recall_score,
                )

                # Log testing metrics
                mlflow.log_metric(
                    "test_f1_score",
                    classification_test_metric.f1_score,
                )
                mlflow.log_metric(
                    "test_precision",
                    classification_test_metric.precision_score,
                )
                mlflow.log_metric(
                    "test_recall",
                    classification_test_metric.recall_score,
                )

                # Log and register the model once

                mlflow.sklearn.log_model(

                    sk_model=best_model,
                    name="model",
                    registered_model_name="NetworkSecurityModel",
                    serialization_format="cloudpickle",
                    )


                logging.info("Model and metrics logged to MLflow successfully.")

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(
                    random_state=42,
                    verbose=1,
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42,
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42,
                    verbose=1,
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    verbose=1,
                ),
                "AdaBoost": AdaBoostClassifier(
                    random_state=42,
                ),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluate and train models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            if not model_report:
                raise ValueError("Model evaluation returned an empty report.")

            # Select the best model
            best_model_name = max(
                model_report,
                key=model_report.get,
            )

            best_model_score = model_report[best_model_name]

            # evaluate_models must fit/update the model objects in 'models'
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")

            # Evaluate training data
            y_train_pred = best_model.predict(X_train)

            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )

            # Evaluate testing data
            y_test_pred = best_model.predict(x_test)

            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )

            # Track metrics and model with MLflow
            self.track_mlflow(
                best_model=best_model,
                classification_train_metric=classification_train_metric,
                classification_test_metric=classification_test_metric,
            )

            # Load the fitted preprocessing object
            preprocessor = load_object(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_object_file_path
                )
            )

            # Prepare the configured model directory
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            if model_dir_path:
                os.makedirs(model_dir_path, exist_ok=True)

            # Bundle preprocessing and trained model
            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )

            # Save the complete inference model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model,
            )

            # Save the classifier separately for compatibility
            # with the existing model-pusher flow
            final_model_path = os.path.join(
                "final_model",
                "model.pkl",
            )

            os.makedirs(
                os.path.dirname(final_model_path),
                exist_ok=True,
            )

            save_object(
                file_path=final_model_path,
                obj=best_model,
            )

            # Create the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

            logging.info(
                f"Model trainer artifact: {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            # Load training and testing arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Separate features and target
            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model_trainer_artifact = self.train_model(
                x_train,
                y_train,
                x_test,
                y_test,
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
