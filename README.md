# CFDnet
domain adaptation with CF distance for medical image segmentation

It is the code for the paper : 'CF Distance: A New Domain Discrepancy Metric and Application to Explicit Domain Adaptation for Cross-Modality Cardiac Image Segmentation', submitted to Transaction on Medical Imaging.

# package requirement
python 2.7\\
SimpleITK 1.2.0
Torch 1.1.0
numpy 1.15.0

# Usage
The core code for CF distance computation is the class `Feature_Distribution_Distance_func' in python file `utils_for_transfer.py'. The mosy important parameter need to be tuned is `self.Tvalue', here we set it to be 1e2. You can initialize it first, and then input any two corresponding features from two domains inot it to compute their distance of CF. As the back propagation for gradient descent can not be done directly by the pytorch, we provided their gradient computation in the class `Feature_Distribution_Distance_Func' in python file `utils_for_transfer.py'. In the total loss, the loss term of CF distance, i.e., Dis_Lamda* CF_Dist, would be good if its initial value be magnitude of 1e2 via choosing suitable Dis_Lamda.

We provided the Unet here as the backbone of the network, you can adjust this for any other network.
