import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MobileNetV3(nn.Module):
    """
    MobileNetV3 Small adaptado para clasificación de imágenes pequeñas (p.ej. CIFAR-10, 32x32).

    Utiliza la arquitectura MobileNetV3 Small de torchvision, ajustando la primera capa y el clasificador final
    para imágenes de 32x32 y 10 clases.

    Args:
        num_classes (int): Número de clases de salida (por defecto 10).
        weights (Optional): Pesos a usar (por defecto None para inicialización aleatoria).
    """

    def __init__(self, num_classes: int = 10, weights=None):
        super().__init__()
        # Cargar MobileNetV3 Small base con inicialización aleatoria o pesos dados
        self.model = mobilenet_v3_small(weights=weights)
        # Ajustar la primera capa para imágenes 32x32 (stride=1)
        self.model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # Ajustar el clasificador final para el número de clases
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_trained_mobilenetv3(path=None, num_classes=10, map_location='cpu'):
    """
    Instancia un modelo MobileNetV3 y carga los pesos entrenados desde el path especificado o desde models_saved/mobilenetv3_cifar10.pt.
    """
    import os
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved')
    if path is None:
        path = os.path.join(data_root, 'mobilenetv3_cifar10.pt')
    model = MobileNetV3(num_classes=num_classes)
    import torch
    state = torch.load(path, map_location=map_location, weights_only=True)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model
