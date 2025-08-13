from pathlib import Path

import torch

from models import make_model, get_default_input_size
from utils.export_checks import try_export_onnx


def test_forward_and_export_all_models(tmp_path: Path):
    models = ["cnn", "mlp", "mobilenetv3", "efficientnet_lite0"]
    for name in models:
        shape = get_default_input_size(name)
        m = make_model(name, num_classes=10)
        m.eval()
        x = torch.randn(*shape)
        with torch.no_grad():
            y = m(x)
        assert y.shape[0] == shape[0]
        assert y.shape[1] == 10
        # Export ONNX
        out = try_export_onnx(m, tmp_path / f"{name}.onnx", sample_shape=shape)
        assert out.exists()
