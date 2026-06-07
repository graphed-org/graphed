"""Triton python-backend model served by the M9 real-Triton CI test (graphed-preserve).

A per-event score = sigmoid(W*x + B). The same closed-form (in numpy float32) is the test's
independent reference, so a bundle reproduced through a REAL Triton server matches bit-for-bit.
The weights W/B are the served model's content; the test content-addresses this file as the payload.
Keep W/B in sync with W/B in tests/frozen/m9/test_triton_server.py.
"""

import numpy as np
import triton_python_backend_utils as pb_utils  # provided by the Triton python backend

W = 0.45
B = -0.1


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            x = pb_utils.get_input_tensor_by_name(request, "x").as_numpy().astype("float32")
            y = (1.0 / (1.0 + np.exp(-(W * x + B)))).astype("float32")
            responses.append(pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("y", y)]))
        return responses
