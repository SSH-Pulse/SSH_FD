# **SSH-FD: Detecting SSH Communication in Tunneled Traffic with Frequency-Domain Representation**

---

## **SSH-FD Overview**

In this project, we introduce **SSH-FD**, an advanced approach for detecting SSH communication within tunneled traffic. Our framework operates in multiple stages to effectively isolate and identify SSH communication, even when encapsulated within other protocols. The process begins by splitting the tunneled traffic into **bursts**—small segments representing transient network behaviors within short time windows. These bursts serve as the primary units for feature extraction, helping to isolate SSH communication from other types of traffic.

To enhance the model's ability to detect subtle differences in tunneled traffic, we perform **spatio-temporal feature processing**. This includes the normalization and distribution optimization of both packet length and interval time, which emphasizes the variations in traffic patterns that are critical for detection. Subsequently, we convert the burst-level features into a **frequency-domain representation**, mapping the raw flow features into frequency domain signals. By aggregating the signals from the packets within a burst, we create a unified and consistent representation that captures key characteristics of the tunneled traffic.

For detecting SSH communication, we utilize an **encoder-decoder model** based on **recurrent neural networks** (RNNs). This model processes the sequence of bursts while preserving the temporal characteristics of the traffic, enabling effective flow-level SSH detection. To balance detection performance at both the burst and flow levels, **multi-scale supervision** is introduced, ensuring that the model maintains sensitivity without being overly influenced by individual burst anomalies.

The overall architecture of **SSH-FD** is depicted in **Fig** below:

![Overall SSH-FD Architecture](https://github.com/user-attachments/assets/4bd070ba-7cd7-4b95-b32c-2779c749c83b)

---

## **Files and Modules**

### **config.py**

The `config.py` module acts as the **centralized configuration hub** for the entire training and evaluation pipeline. It contains all the critical hyperparameters, architectural settings, dataset paths, logging configurations, and feature selection parameters in a structured and readable format. This file serves as the foundation for configuring various components, ensuring flexibility during experimentation.

### **model.py**

The `model.py` module defines the complete **Seq2Seq model with attention**, central to the architecture of SSH-FD. It comprises the following components:

* **Encoder**: A bidirectional LSTM-based encoder that processes the input sequence.
* **Attention**: An additive attention mechanism inspired by Bahdanau et al., which enables the model to focus on relevant parts of the input sequence.
* **Decoder**: An attentive LSTM-based decoder that generates the output sequence by combining embeddings and context information from the attention mechanism.

### **npy\_dataloader.py**

The `npy_dataloader.py` module handles the **feature extraction and data loading** process. It is responsible for:

* Extracting multiple **statistical** and **signal-domain features** from packet lengths and timestamps.
* Dynamically selecting feature combinations based on the specific experiment or task.
* Generating and saving structured `.npy` feature arrays from raw **JSON data**.
* Preparing PyTorch **DataLoader objects** for both training and evaluation, ensuring efficient and seamless data handling during model training.

### **SSH\_FD\_main.py**

The `SSH_FD_main.py` script orchestrates the **training pipeline** for the Seq2Seq model applied to **burst detection** and **flow detection** tasks. Key components of the script include:

* **Encoder and Decoder Models**: Defines the architecture of the sequence-to-sequence model.
* **Training Pipeline**: Manages the training loop, validation process, and **early stopping** mechanisms to prevent overfitting.
* **Feature Combination**: Dynamically handles different feature combinations, enabling the model to adapt to various configurations.
* **Configuration and Logging**: The script integrates with the `config.py` file for configuration management and supports detailed logging of key events throughout the training process.

### **train.py**

The `train.py` script is dedicated to **training and evaluating** the deep learning model for burst and flow detection. It incorporates the following key features:

* **Optimization with Adam**: The model is optimized using the Adam optimizer for stable and efficient training.
* **Multiple Loss Functions**: The script incorporates **binary cross-entropy losses** for both burst and flow detection tasks.
* **Gradient Clipping**: To prevent gradient explosion, gradient clipping is employed during training.
* **Early Stopping**: Implements early stopping to halt training when performance ceases to improve, preventing overfitting.
* **Evaluation Metrics**: Utility functions for evaluating the model's performance using **F1-score, precision, recall**, and **accuracy** for both burst and flow detection tasks.

### **util.py**

The `util.py` script provides essential **utility functions** for managing GPU memory and setting up logging systems:

* **GPU Memory Management**: Clears unused GPU memory during training to prevent memory overflow.
* **Logger Setup**: Configures a logging system to capture important events and track progress throughout the training and evaluation processes.

---

## **Dataset**

The dataset used for training and evaluation contains network packet data and can be downloaded from the following Google Drive link:

[Download Dataset](https://drive.google.com/drive/folders/1JlgmUIC8oXVzso7a19gFrC8uZk1A5Kj4?usp=drive_link)

The dataset consists of features such as packet lengths and timestamps, which are processed to extract relevant statistical and signal-domain features for model training.

---

## **Installation**

### Prerequisites

To run the code, the following Python libraries are required:

* Python 3.x
* PyTorch (recommended version 1.9.0 or later)
* NumPy
* Pandas
* scikit-learn
* tqdm

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/SSH-Pulse/SSH_FD.git
   cd SSH_FD
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### **Usage**


To train the model, run the `SSH_FD_main.py` script. It will automatically load the configuration from `config.py` and start the training process.

```bash
python SSH_FD_main.py

