import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "tabular_model.pth")

        instance_kind = args.get("model_instance_kind", "cpu").lower()
        if instance_kind == "gpu":
            device_id = int(args.get("model_instance_device_id", 0))
            torch.cuda.set_device(device_id)
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        # Optional: If your model outputs class labels
        self.classes = np.array(["low", "medium", "high"])  # Example class labels

    def execute(self, requests):
        # Collect and concatenate inputs from all requests
        all_ages = []
        all_incomes = []
        all_genders = []
        request_batch_sizes = []

        for request in requests:
            age_tensor = pb_utils.get_input_tensor_by_name(request, "age")
            income_tensor = pb_utils.get_input_tensor_by_name(request, "income")
            gender_tensor = pb_utils.get_input_tensor_by_name(request, "gender")

            age = age_tensor.as_numpy().astype(np.float32)  # shape [B,1]
            income = income_tensor.as_numpy().astype(np.float32)
            gender = gender_tensor.as_numpy()  # shape [B,1], dtype object (BYTES)

            batch_size = age.shape[0]
            request_batch_sizes.append(batch_size)

            # Decode gender strings and encode to float32 (e.g., male=0, female=1, other=2)
            gender_strs = np.vectorize(lambda b: b.decode("utf-8"))(gender)
            gender_encoded = np.where(gender_strs == "male", 0,
                               np.where(gender_strs == "female", 1, 2)).astype(np.float32)

            all_ages.append(age)
            all_incomes.append(income)
            all_genders.append(gender_encoded)

        # Stack all batches into one big batch
        ages = np.vstack(all_ages)          # shape [total_batch, 1]
        incomes = np.vstack(all_incomes)
        genders = np.vstack(all_genders)

        # Final feature tensor: shape [total_batch, 3]
        features = np.concatenate([ages, incomes, genders], axis=1)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(features_tensor)  # [total_batch, num_classes]

        responses = []
        start = 0
        for batch_size in request_batch_sizes:
            end = start + batch_size
            batch_outputs = outputs[start:end]

            for output in batch_outputs:
                if self.classes is not None and output.shape[0] > 1:
                    prob = torch.softmax(output, dim=0)
                    pred_idx = torch.argmax(prob).item()
                    pred_label = self.classes[pred_idx]
                    out_np = np.array([[pred_label]], dtype=object)
                    out_tensor = pb_utils.Tensor("PRED_LABEL", out_np)
                else:
                    val = output.item()
                    out_np = np.array([[val]], dtype=np.float32)
                    out_tensor = pb_utils.Tensor("PRED_VALUE", out_np)

                response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(response)

            start = end

        return responses


