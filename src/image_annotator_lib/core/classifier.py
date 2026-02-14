"""
Classifier module for CLIP-based image scoring models.

CLIP ベースの画像スコアリングモデル用の分類器。
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Flexible classifier taking image features as input and outputting classification scores.
    / 画像特徴量を入力として、分類スコアを出力する柔軟な分類器。

    Allows configurable hidden layers, dropout rates, and activation functions.
    / 設定可能な隠れ層、ドロップアウト率、活性化関数を持つ。

    Args:
        input_size: Dimension of the input features.
        hidden_sizes: List of sizes for the hidden layers. Defaults to [1024, 128, 64, 16].
        output_size: Dimension of the output. Defaults to 1.
        dropout_rates: List of dropout rates for each hidden layer. Defaults to match hidden_sizes.
        use_activation: Whether to use activation functions in hidden layers. Defaults to False.
        activation: Activation function type for hidden layers. Defaults to nn.ReLU.
        use_final_activation: Whether to use an activation function on the final layer. Defaults to False.
        final_activation: Activation function type for the final layer. Defaults to nn.Sigmoid.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        output_size: int = 1,
        dropout_rates: list[float] | None = None,
        use_activation: bool = False,
        activation: type[nn.Module] = nn.ReLU,
        use_final_activation: bool = False,
        final_activation: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes if hidden_sizes is not None else [1024, 128, 64, 16]
        dropout_rates = dropout_rates if dropout_rates is not None else [0.2, 0.2, 0.1, 0.0]
        if len(dropout_rates) < len(hidden_sizes):
            dropout_rates.extend([0.0] * (len(hidden_sizes) - len(dropout_rates)))

        layers: list[nn.Module] = []
        prev_size = input_size
        for size, drop in zip(hidden_sizes, dropout_rates, strict=False):
            layers.append(nn.Linear(prev_size, size))
            if use_activation:
                layers.append(activation())
            if drop > 0:
                layers.append(nn.Dropout(drop))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        if use_final_activation:
            layers.append(final_activation())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the classifier layers.
        / 分類器レイヤーを通してフォワードパスを実行する。
        """
        return self.layers(x)
