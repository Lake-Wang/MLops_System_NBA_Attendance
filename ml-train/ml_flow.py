import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Let's get results of model 1
experiment = client.get_experiment_by_name("model1")

runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
    order_by=["metrics.test_rmse ASC"], 
    max_results=2)

best_run = runs[0]  # The first run is the best due to sorting
best_run_id = best_run.info.run_id
best_test_accuracy = best_run.data.metrics["test_rmse"]
model_uri = f"runs:/{best_run_id}/model"

print(f"Best Run ID: {best_run_id}")
print(f"Test Accuracy: {best_test_accuracy}")
print(f"Model URI: {model_uri}")

model_name = "model1-staging"
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model registered as '{model_name}', version {registered_model.version}")

model_versions = client.search_model_versions(f"name='{model_name}'")

latest_version = max(model_versions, key=lambda v: int(v.version))

print(f"Latest registered version: {latest_version.version}")
print(f"Model Source: {latest_version.source}")
print(f"Status: {latest_version.status}")

# This would need to go to block storage, but would go in GitHub for now
local_download = mlflow.artifacts.download_artifacts(latest_version.source, dst_path="./downloaded_model1")

## Let's get results of model 2
experiment = client.get_experiment_by_name("model2")

runs = client.search_runs(experiment_ids=[experiment.experiment_id], 
    order_by=["metrics.test_rmse ASC"], 
    max_results=2)

best_run = runs[0]  # The first run is the best due to sorting
best_run_id = best_run.info.run_id
best_test_accuracy = best_run.data.metrics["test_rmse"]
model_uri = f"runs:/{best_run_id}/model"

print(f"Best Run ID: {best_run_id}")
print(f"Test Accuracy: {best_test_accuracy}")
print(f"Model URI: {model_uri}")

# Staging piece
model_name = "model2-staging"
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model registered as '{model_name}', version {registered_model.version}")

model_versions = client.search_model_versions(f"name='{model_name}'")

latest_version = max(model_versions, key=lambda v: int(v.version))

print(f"Latest registered version: {latest_version.version}")
print(f"Model Source: {latest_version.source}")
print(f"Status: {latest_version.status}")

local_download = mlflow.artifacts.download_artifacts(latest_version.source, dst_path="./downloaded_model2")