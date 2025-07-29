import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    CNN (Convolutional Neural Network) para clasificación de imágenes pequeñas (p.ej. CIFAR-10, FashionMNIST).

    Arquitectura típica:
        Conv2d → ReLU → (BatchNorm2d) → MaxPool2d → Conv2d → ReLU → (BatchNorm2d) → MaxPool2d → Flatten → Linear → ReLU → Linear (output)

    Args:
        input_channels (int): Número de canales de entrada (1 para FashionMNIST, 3 para CIFAR-10).
        num_classes (int): Número de clases de salida (por defecto 10).
        input_size (int): Tamaño de la imagen de entrada (por defecto 28 o 32).
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 10, input_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        # Calcular tamaño de entrada para la capa lineal
        fc_input_size = 64 * (input_size // 4) * (input_size // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_trained_cnn(path=None, input_channels=3, num_classes=10, input_size=32, map_location='cpu'):
    """
    Instancia un modelo CNN y carga los pesos entrenados desde el path especificado o desde models_saved/cnn_cifar10.pt.
    """
    import os
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved')
    if path is None:
        path = os.path.join(data_root, 'cnn_cifar10.pt')
    model = CNN(input_channels=input_channels, num_classes=num_classes, input_size=input_size)
    import torch
    state = torch.load(path, map_location=map_location, weights_only=True)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model
