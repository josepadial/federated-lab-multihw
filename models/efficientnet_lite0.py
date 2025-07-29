import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetLite0(nn.Module):
    """
    EfficientNet-lite0 adaptado para clasificación de imágenes pequeñas (CIFAR-10, 32x32).

    Utiliza la arquitectura EfficientNet-B0 de torchvision como base, ajustando la primera capa y el clasificador final
    para imágenes de 32x32 y 10 clases. Es una alternativa ligera y eficiente para despliegue en edge/federated learning.

    Args:
        num_classes (int): Número de clases de salida (por defecto 10).
        pretrained (bool): Si cargar pesos preentrenados en ImageNet (por defecto False).
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        # Cargar EfficientNet-B0 base
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = efficientnet_b0(weights=weights)
        # Ajustar la primera capa para imágenes 32x32 (stride=1)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # Ajustar el clasificador final para el número de clases
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_trained_efficientnetlite0(path=None, num_classes=10, pretrained=False, map_location='cpu'):
    """
    Instancia un modelo EfficientNetLite0 y carga los pesos entrenados desde el path especificado o desde models_saved/efficientnetlite0_cifar10.pt.
    """
    import os
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved')
    if path is None:
        path = os.path.join(data_root, 'efficientnetlite0_cifar10.pt')
    model = EfficientNetLite0(num_classes=num_classes, pretrained=pretrained)
    import torch
    state = torch.load(path, map_location=map_location, weights_only=True)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model
