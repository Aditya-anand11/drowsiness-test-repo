import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)

class ResNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block1 = ResidualBlock(32, 32)
        self.block2 = ResidualBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.fc(x))

class InceptionBlock(nn.Module):
    def __init__(self, in_c, f1, f2, f3):
        super().__init__()
        self.b1 = nn.Conv2d(in_c, f1, kernel_size=1)
        self.b2 = nn.Conv2d(in_c, f2, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(in_c, f3, kernel_size=5, padding=2)
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_c, f1, kernel_size=1)
        )

    def forward(self, x):
        out1 = F.relu(self.b1(x))
        out2 = F.relu(self.b2(x))
        out3 = F.relu(self.b3(x))
        out4 = F.relu(self.b4(x))
        return torch.cat([out1, out2, out3, out4], dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.inc = InceptionBlock(32, 16, 16, 16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.inc(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.fc(x))

class CNN2D(nn.Module):
    def __init__(self, use_bn=True, dropout=0.5, init_type="he"):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 1)

        self._init_weights(init_type)

    def _init_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class CNN3D(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1(x)), kernel_size=(1, 2, 2))
        x = F.max_pool3d(F.relu(self.conv2(x)), kernel_size=(2, 2, 2))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.fc(x))

class YOLOHead(nn.Module):
    def __init__(self, in_c=64, num_classes=3, num_anchors=2):
        super().__init__()
        self.head = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        return self.head(x)

def focal_loss(preds, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy(preds, targets, reduction="none")
    p_t = preds * targets + (1 - preds) * (1 - targets)
    loss = bce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).mean()
