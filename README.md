# 🎨 Image Colorization with Conditional Wasserstein GANs

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-1.5%2B-purple.svg)](https://www.pytorchlightning.ai/)

> **A deep learning approach to automatic image colorization using Conditional Wasserstein GANs with advanced regularization techniques.**


## 🌟 Project Overview

This project implements a **Conditional Wasserstein GAN (CWGAN)** architecture for automatic grayscale-to-color image translation. The system leverages advanced deep learning techniques including gradient penalty, R1 regularization, and a ResU-Net generator architecture to produce high-quality, realistic colorizations.

### 🎯 Key Features

- **GAN Architecture**: Conditional Wasserstein GAN with gradient penalty and R1 regularization
- **Custom ResU-Net Generator**: U-Net with residual connections for enhanced feature preservation
- **PatchGAN Discriminator**: Efficient adversarial training with instance normalization
- **LAB Color Space**: Perceptually uniform color representation for optimal results
- **Evaluation**: Inception Score and FID metrics for quantitative assessment


## 🏗️ Architecture

### Generator: ResU-Net
```
📦 ResU-Net Generator (8.2M parameters)
├── Encoder Path
│   ├── ResBlock(1→64) + Dropout
│   ├── DownSample(64→128) + Dropout  
│   ├── DownSample(128→256) + Dropout
│   └── Bridge(256→512) + Dropout
└── Decoder Path
    ├── UpSample(512→256) + Skip Connection
    ├── UpSample(256→128) + Skip Connection
    ├── UpSample(128→64) + Skip Connection
    └── Output Conv(64→2)
```

### Discriminator: PatchGAN Critic
```
🔍 PatchGAN Critic (2.7M parameters)
├── Conv2d(3→64) + LeakyReLU
├── Conv2d(64→128) + InstanceNorm + LeakyReLU
├── Conv2d(128→256) + InstanceNorm + LeakyReLU
├── Conv2d(256→512) + InstanceNorm + LeakyReLU
├── AdaptiveAvgPool2d(1)
└── Linear(512→1)
```

## 📊 Results & Performance

### Quantitative Metrics
| Metric | Real Images | Generated Images |
|--------|-------------|------------------|
| **Inception Score** | 4.34 ± 1.80 | 4.31 ± 1.84 |
| **FID Score** | - | **13.11** |

### Training Configuration
- **Optimizer**: Adam (β₁=0.5, β₂=0.9)
- **Learning Rate**: 2e-4
- **Loss Function**: L1 Reconstruction + WGAN-GP + R1 Regularization
- **Batch Size**: 1 (GPU memory optimized)
- **Epochs**: 150
- **Regularization**: λ_GP=10, λ_R1=10, λ_recon=100

## 📊 Dataset

The model is trained on the **Image Colorization Dataset**:

🔗 **Dataset Source**: [Kaggle - Image Colorization Dataset](https://www.kaggle.com/datasets/shravankumar9892/image-colorization)

- **Training samples**: 5,000 images (224×224 resolution)
- **Format**: Separate `.npy` files for L and AB channels
- **Preprocessing**: Images converted to LAB color space with normalization

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision pytorch-lightning
pip install opencv-python scikit-image matplotlib
pip install numpy pandas tqdm torchsummary
```

### Usage
1. **Data Preparation**: Download dataset and convert images to LAB color space
```python
# Download from Kaggle: https://www.kaggle.com/datasets/shravankumar9892/image-colorization
# L channel (grayscale) as input - shape: (224,224,1)
# AB channels (color information) as ground truth - shape: (224,224,2)
```

2. **Training**:
```python
cwgan = CWGAN(in_channels=1, out_channels=2, learning_rate=2e-4)
trainer = pl.Trainer(max_epochs=150, accelerator="gpu")
trainer.fit(cwgan, train_loader)
```

3. **Inference**:
```python
# Generate colorized image from grayscale input
colorized = cwgan.generator(grayscale_image)
```

## 🔬 Technical Highlights

### Advanced Training Techniques
- **Wasserstein Loss with Gradient Penalty**: Stable GAN training with improved convergence
- **R1 Regularization**: Enhanced discriminator stability and reduced mode collapse
- **LAB Color Space**: Perceptually uniform representation for better color fidelity
- **Residual Connections**: Gradient flow optimization in deep U-Net architecture

### Model Innovations
- **Skip Connections**: Preserve fine-grained details during upsampling
- **Instance Normalization**: Improved discriminator performance
- **Adaptive Pooling**: Resolution-independent feature extraction
- **Progressive Training**: Stable adversarial learning dynamics

## 📁 Project Structure

```
Image Colorization using Conditional Wasserstein GANs/
├── image-colorization-cwgan.ipynb    # Main implementation
├── Project_Report.pdf                # Technical documentation
├── Image Colorization using Conditional Wasserstein GANs.pptx  # Results presentation
└── README.md                         # Project documentation
```

## 🛠️ Implementation Details

### Dataset Processing
- **Input**: Grayscale images (L channel) - 224×224×1
- **Output**: Color channels (AB channels) - 224×224×2
- **Preprocessing**: LAB color space conversion with normalization
- **Data Augmentation**: Standard computer vision transforms

### Loss Functions
```python
# Generator Loss
L_gen = L1_loss(fake_ab, real_ab) + λ_adv * WGAN_loss

# Discriminator Loss  
L_disc = WGAN_loss + λ_gp * gradient_penalty + λ_r1 * r1_regularization
```

## 📈 Performance Analysis

The model achieves competitive results with:
- **High Visual Fidelity**: Realistic color distributions matching natural images
- **Stable Training**: Wasserstein distance provides stable gradient signals  
- **Efficient Architecture**: Optimized parameter count vs. performance ratio
- **Quantitative Validation**: Professional evaluation metrics (IS, FID)



## 📝 Future Enhancements

- [ ] Multi-scale discriminator for better detail preservation
- [ ] Perceptual loss integration for enhanced realism
- [ ] Real-time inference optimization
- [ ] Web interface for interactive colorization
- [ ] Dataset expansion with diverse image categories

## 📚 References

[1] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2014, pp. 2672–2680.

[2] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 1125–1134.

[3] M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein generative adversarial networks," in Proc. Int. Conf. Mach. Learn. (ICML), 2017, pp. 214–223.

[4] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, "Improved training of Wasserstein GANs," in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2017, pp. 5767–5777.

[5] O. Ronneberger, P. Fischer, and T. Brox, "U-net: Convolutional networks for biomedical image segmentation," in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2015, pp. 234–241.

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.

[7] R. Zhang, P. Isola, and A. A. Efros, "Colorful image colorization," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016, pp. 649–666.

[8] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, "Improved techniques for training GANs," in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2016, pp. 2234–2242.

## 👤 Author

**Sohan Arun**  
Master’s Student, Computer Science  
Blekinge Institute of Technology, Sweden  
📧 [Sohanoffice46@gmail.com](mailto:Sohanoffice46@gmail.com)

For detailed technical analysis, experimental setup, and theoretical background, please refer to the complete research report: Project_Report.pdf
