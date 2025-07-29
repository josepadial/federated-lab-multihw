import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    MLP (Multilayer Perceptron) para clasificación de imágenes pequeñas (p.ej. CIFAR-10, FashionMNIST).

    Arquitectura:
        Flatten → Linear → ReLU → Linear → ReLU → Linear (output)

    Args:
        input_size (int): Número de características de entrada (por defecto 28*28 para FashionMNIST, 32*32*3 para CIFAR-10).
        num_classes (int): Número de clases de salida (por defecto 10).
        hidden1 (int): Unidades en la primera capa oculta (por defecto 256).
        hidden2 (int): Unidades en la segunda capa oculta (por defecto 128).
    """

    def __init__(self, input_size: int = 28 * 28, num_classes: int = 10, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def load_trained_mlp(path=None, input_size=32 * 32 * 3, num_classes=10, map_location='cpu'):
    """
    Instancia un modelo MLP y carga los pesos entrenados desde el path especificado o desde models_saved/mlp_cifar10.pt.
    """
    import os
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models_saved')
    if path is None:
        path = os.path.join(data_root, 'mlp_cifar10.pt')
    model = MLP(input_size=input_size, num_classes=num_classes)
    import torch

    state = torch.load(path, map_location=map_location, weights_only=True)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model
