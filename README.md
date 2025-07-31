# DeepDICI

## Requirements
- Python 3.7+
- PyTorch >= 1.7
- numpy
- scikit-learn
- tqdm
- tensorboardX
- prefetch_generator

You can install the dependencies with:
```bash
pip install torch numpy scikit-learn tqdm tensorboardX prefetch_generator
```

## Quick Start
1. **Prepare your data**
   - Place your processed data files (e.g., `ionchannel_dataset/proteins_dict.pkl`, `embedding/drug_dict.pkl`) in the `input/` directory under the project root.

2. **Train and Evaluate**
   - Run the main script:
   ```bash
   python main.py
   ```
   - Training and evaluation results will be saved in the `output/` directory.

## Main Files
- `main.py`: Main training and evaluation script.
- `model.py`: Contains the `DeepDICI` model definition.
- `dataset.py`: Data loading and preprocessing utilities.
- `hyperparameter.py`: Hyperparameter configuration.
- `pytorchtools.py`: Early stopping and training utilities.

## Project Structure
```
DeepDICI/
├── data/           # Input data directory
├── result          # Output results directory
├── model.py              # Model definition
├── dataset.py            # Dataset and dataloader
├── hyperparameter.py     # Hyperparameters
├── main.py       # Main script
├── pytorchtools.py       # Training utilities
└── ...
```

## Training and Testing
- The script supports 5-fold cross-validation by default.
- Results for each fold and overall statistics are saved in the output directory.
- Training progress and metrics can be visualized using TensorBoard:
  ```bash
  tensorboard --logdir output/
  ```
