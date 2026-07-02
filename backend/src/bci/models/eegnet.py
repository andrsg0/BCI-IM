"""EEGNet — la red neuronal compacta para EEG (Lawhern et al., 2018).

POR QUÉ ESTÁ AQUÍ (el puente LTI ↔ IA)
--------------------------------------
EEGNet es la parte de "IA" del proyecto, y se justifica porque sus capas
**imitan el pipeline clásico de filtros LTI**, pero APRENDIÉNDOLOS de los datos:

  1. Conv2D temporal (1×kernLength)  ≈ un BANCO DE FILTROS FIR aprendidos
     (cada filtro es una respuesta al impulso que la red ajusta).
  2. DepthwiseConv2D espacial (C×1)  ≈ un filtro espacial tipo CSP aprendido
     (combina los canales, como W·X).
  3. SeparableConv2D + pooling       ≈ extracción de potencia por banda.

Así, en vez de DISEÑAR el FIR (banda µ/β) y CALCULAR el CSP, la red los
descubre por sí misma. Más adelante visualizaremos esos filtros aprendidos
para comprobar que, efectivamente, redescubre la banda µ/β y la lateralización
motora.

Se entrena con PyTorch (GPU si está disponible, si no CPU). El wrapper
EEGNetClassifier ofrece fit/predict al estilo sklearn para integrarse con el
resto del pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def pick_device(device: str | None = None) -> str:
    """Elige el dispositivo de cómputo. ``None`` ⇒ 'cuda' si hay GPU, si no 'cpu'."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class EEGNet(nn.Module):
    """Arquitectura EEGNet. Entrada: (batch, 1, n_canales, n_muestras)."""

    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kern_length: int = 64, dropout: float = 0.25):
        super().__init__()
        # Bloque 1: filtro temporal (FIR aprendido) + filtro espacial (CSP aprendido)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise por canal
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Bloque 2: convolución separable (depthwise + pointwise) = potencia por banda
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # tamaño aplanado: lo calculamos con un forward de prueba (robusto al padding)
        with torch.no_grad():
            n_feat = self._embed(torch.zeros(1, 1, n_channels, n_samples)).shape[1]
        self.classifier = nn.Linear(n_feat, n_classes)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._embed(x))


class EEGNetClassifier:
    """Wrapper estilo sklearn: estandariza, entrena con Adam y predice."""

    def __init__(self, n_classes: int = 2, epochs: int = 250, lr: float = 1e-3,
                 batch_size: int = 32, kern_length: int = 64, dropout: float = 0.25,
                 weight_decay: float = 0.0, seed: int = 42, device: str | None = None):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.kern_length = kern_length
        self.dropout = dropout
        self.weight_decay = weight_decay   # regularización L2 (filtros más suaves)
        self.seed = seed
        self.device = pick_device(device)   # None ⇒ GPU si está disponible
        self.n_classes = n_classes
        self.model: EEGNet | None = None
        self.classes_: np.ndarray | None = None
        self.mean_ = None
        self.std_ = None

    def __setstate__(self, state: dict) -> None:
        """Al DESERIALIZAR, remapea el modelo al dispositivo disponible.

        El .pkl pudo guardarse en una máquina con GPU (``self.device == 'cuda'`` y pesos
        en CUDA). Los storages ya se cargaron a CPU (ver ``_CPUUnpickler`` en
        ``pipeline.training``); aquí ajustamos el atributo ``device`` al que HAYA en esta
        máquina y movemos allí los pesos, para que ``_prep``/``predict`` no intenten usar
        un 'cuda' inexistente. Así el mismo artefacto sirve en CPU y en GPU.
        """
        self.__dict__.update(state)
        self.device = pick_device(None)
        if getattr(self, "model", None) is not None:
            self.model.to(self.device)

    def _prep(self, X: np.ndarray) -> torch.Tensor:
        # estandariza por canal con la media/desviación del entrenamiento
        Xs = (X - self.mean_) / self.std_
        t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        return t.unsqueeze(1)  # (n, 1, C, T)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)

        # estandarización por canal (media/std sobre trials y tiempo)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-7

        self.model = EEGNet(X.shape[1], X.shape[2], len(self.classes_),
                            kern_length=self.kern_length, dropout=self.dropout).to(self.device)
        Xt = self._prep(X)
        yt = torch.tensor(y_idx, dtype=torch.long, device=self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        n = len(Xt)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = self.model(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        logits = self.model(self._prep(np.asarray(X, dtype=np.float32)))
        idx = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[idx]

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades por clase (softmax de los logits). Útil para confianza/voto."""
        self.model.eval()
        logits = self.model(self._prep(np.asarray(X, dtype=np.float32)))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
