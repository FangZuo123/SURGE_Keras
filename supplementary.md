# SURGE: Self-supervised Upsampling for Reconstructions with Generalizable Enhancement in Photoacoustic Computed Tomography

## Table of Contents
1. [Model Description](#method-description)
2. [Up-sampling Procedure Description](#method-description)
3. [Experimental Results](#experimental-results)
4. [Noise Test Results](#noise-test-dynamic-results)




## Model Description
### 1. **U-Net**
U-Net is a convolutional neural network (CNN) architecture originally designed for biomedical image segmentation tasks. Its key feature is the symmetric encoder-decoder structure, which allows for efficient feature extraction and spatial information recovery. The architecture’s main advantage is the use of **skip connections** between the encoder and decoder, which help preserve fine-grained details, making it particularly effective for image reconstruction tasks.

- **Encoder**: A series of convolutional layers and pooling operations gradually extract low-level features from the input image.
- **Bottleneck**: After several convolutions and activations, the high-level features of the image are captured.
- **Decoder**: A series of deconvolution (upsampling) layers gradually recover the spatial resolution of the image.
- **Skip Connections**: Skip connections pass low-level feature maps directly from the encoder to the corresponding layers in the decoder, ensuring important spatial information is retained.

The U-Net architecture is especially well-suited for image segmentation and reconstruction tasks where preserving detailed features is crucial, such as in medical imaging and remote sensing.

### 2. **SRResNet**
SRResNet is a deep convolutional neural network architecture designed specifically for image super-resolution tasks. Unlike traditional super-resolution methods, SRResNet employs **residual learning** to optimize the training process, enabling it to effectively recover high-frequency details and generate high-quality super-resolved images.

- **Residual Learning**: SRResNet uses **Residual Blocks**, which introduce skip connections to bypass layers, improving gradient flow and helping to avoid the vanishing gradient problem during training.
- **Deep Network Structure**: The architecture consists of several residual blocks that help capture detailed features at multiple levels.
- **Activation Function**: **ReLU** activations and **batch normalization** are used to improve the stability and convergence of the training process.

SRResNet has been shown to perform exceptionally well in image super-resolution tasks, especially in recovering fine details that are critical for high-quality image reconstruction.

### Reference Papers
For detailed information on the U-Net and SRResNet architectures, please refer to the original papers:

- **U-Net**: [Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **SRResNet**: [Ledig et al., Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

### Training Details
- **Batch Size**: 8
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: L1 + L0

## Experimental Results
### Image Reconstruction Comparison
## Results

To evaluate the performance of the SURGE method in a real-world scenario, we applied it to enhance the image quality of photoacoustic tomography data from our previous study: [Wang X, Luo Y, Chen Y, et al. A skull-removed chronic cranial window for ultrasound and photoacoustic imaging of the rodent brain. Frontiers in Neuroscience, 2021, 15: 673740](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.673740/full). In this study, although a 512-element ring array would have been ideal, only 256 channels were used due to experimental limitations.

We applied the SURGE method to upsample the original 256-channel data. The results demonstrate a significant improvement in image quality compared to conventional linear interpolation. Specifically, the original sparse sampling (256 channels), linear interpolation, and SURGE upsampling (to 512 channels) are compared in the figure below.

From left to right, the sub-panels represent:
1. **Sparse Sampling**: Based on 256 channels, showing prominent streak artifacts.
2. **Linear Interpolation**: Upsampled to 512 channels but still retains significant artifacts.
3. **SURGE Method**: Our method, which significantly reduces the streak artifacts and provides a much clearer reconstruction.

The regions of interest (highlighted with blue, pink, and orange boxes) and their corresponding enlarged images demonstrate how the SURGE method effectively mitigates the artifacts that persist in the linear interpolation result.

![Sparse Sampling, Linear Interpolation, and SURGE Results](Wang_Compare.jpg)
*Figure 1: Comparison of reconstruction quality between SURGE and traditional methods.*


## Noise Test Dynamic Results
Here is a GIF showing how the model performs on noisy data over time. The noise reduction is demonstrated as the model progressively enhances the image quality.
![Noise Test GIF](path/to/your_noise_test.gif)

## Installation and Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/SURGE_Keras.git
