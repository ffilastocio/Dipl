import torch
import torch.nn as nn
import models
from torchvision.models.vision_transformer import vit_b_16

class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(512)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.batchnorm1(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.dropout1(x)
        x = self.batchnorm2(self.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.batchnorm3(self.relu(self.conv5(x)))
        x = self.dropout2(x)
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(self.flatten(x))))))
        return x
    
class DeepCNNMinReg(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCNNMinReg, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(1024)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(65536, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.batchnorm1(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.batchnorm2(self.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.batchnorm3(self.relu(self.conv6(x)))
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(self.flatten(x))))))
        return x
    
class CNN(nn.Module):
    class BNReLUConv2d(nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=True, relu=True, bias=True):
            super(CNN.BNReLUConv2d, self).__init__()      
            if batchnorm: self.append(nn.BatchNorm2d(in_channels))
            if relu: self.append(nn.ReLU())
            self.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

    def __init__(self, num_kernels=[256, 256, 256, 128, 128]):
        super(CNN, self).__init__()
        assert len(num_kernels) == 5
        self.cnns = nn.Sequential()
        for i in range(len(num_kernels)):
            if i == 0:
                self.cnns.append(self.BNReLUConv2d(3, num_kernels[i], 4, 2, 1, batchnorm=False, relu=False, bias=False))
            else:
                kernel_size = 4 if i != len(num_kernels) - 1 else 2
                stride = 2 if i != len(num_kernels) - 1 else 1
                bias = False if i != len(num_kernels) - 1 else True
                padding = 1 if i != len(num_kernels) - 1 else 0
                self.cnns.append(self.BNReLUConv2d(num_kernels[i-1], num_kernels[i], kernel_size, stride, padding, batchnorm=True, relu=True, bias=bias))
        self.apply(CNN._initialize_weights)

    def forward(self, x): return self.cnns(x).squeeze(-1).squeeze(-1)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.batchnorm1(self.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(self.flatten(x))))))
        return x
    
    def extract_features(self, x):
        x = self.batchnorm1(self.relu(self.conv1(x)))
        x = self.pool(x)
        return x

class Simplex2NN(nn.Module):
    def __init__(self):
        super(Simplex2NN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 16 * 16, self.num_classes)  # Assuming input images are 32x32

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x
    
class TriggerSensitiveCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TriggerSensitiveCNN, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16384 , 512)  # Assuming input images are 32x32
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
import torch
import torch.nn as nn
import torchvision.models as models

class TriggerSensitiveVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(TriggerSensitiveVGG, self).__init__()
        self.num_classes = num_classes

        # Load the pretrained VGG16 model
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Modify the feature extractor to handle 32x32 images by removing the last maxpool layer
        self.feature_extractor = nn.Sequential(*list(vgg16.features.children())[:-1])

        # Define new fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 2 * 2, 512)  # Output size is 512x2x2 after the modified feature extractor
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        # Pass input through the VGG16 feature extractor
        x = self.feature_extractor(x)

        # Flatten the output from the feature extractor
        x = self.flatten(x)

        # Pass through the custom fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def extract_features(self, x):
        # Extract features only
        with torch.no_grad():
            return self.feature_extractor(x)
        
class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, 10)
    def forward(self, x):
        return self.vit(x)
    

class PretrainedVGG16Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        
        super(PretrainedVGG16Autoencoder, self).__init__()
        # Pretrained VGG16 as the encoder
        vgg16 = models.vgg16(pretrained=True)
        self.encoder = nn.Sequential(*list(vgg16.features.children())[:31])  # Use up to the last convolutional layer
        
        # Flatten and reduce to latent space
        self.flatten = nn.Flatten()
        self.fc_latent = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, latent_dim)
        )
        
        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512 * 7 * 7),
            nn.ReLU()
        )
        self.unflatten = nn.Unflatten(1, (512, 7, 7))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28 -> 56x56
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x):
        # Encoding
        features = self.encoder(x)
        flattened = self.flatten(features)
        latent = self.fc_latent(flattened)
        
        # Decoding
        x = self.fc_decoder(latent)
        x = self.unflatten(x)
        reconstructed = self.decoder(x)
        return reconstructed, latent
        
