# Skin Cancer Classification with Deep Learning


This project aims to classify skin cancer images using a convolutional neural network (CNN). The dataset used consists of skin lesion images, and the model predicts the likelihood of various types of skin cancer.

## Dataset

The dataset used in this project is from [Kaggle’s Skin Cancer dataset](https://www.kaggle.com/datasets/). The dataset contains labeled images of skin lesions, and the task is to classify the images into one of the categories.

### Dataset Structure
- `dataset/`: Folder containing the dataset images.
    - `train/`: Contains images for training.
    - `test/`: Contains images for testing.
  
## Project Details

### Files:
- **`notebook/skin_cancer_classification.ipynb`**: Jupyter notebook containing the deep learning model and analysis.
- **`dataset/`**: Contains the images used for training and testing the model.
- **`Original notebook/`**: Can be found in kaggle - https://www.kaggle.com/code/jaroslavrutkovskij/skin-cancer-classification

### Libraries Used:
- `TensorFlow` / `Keras` for building and training the neural network.
- `Matplotlib` and `Seaborn` for visualization.
- `NumPy` and `Pandas` for data manipulation.

## Instructions

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/username/skin-cancer-classification.git
    cd skin-cancer-classification
    ```

2. **Set up Environment:**
   It's recommended to set up a virtual environment to avoid conflicts with other projects.
   - Use `venv` or `conda` to create an environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Notebook:**
   Open the `skin_cancer_classification.ipynb` notebook in Jupyter or Google Colab.

5. **Train and Evaluate the Model:**
   Follow the instructions in the notebook to train and evaluate the model on the dataset.

## Results
The model achieves an accuracy of **X%** on the validation dataset, and further improvements can be made through hyperparameter tuning or by using more advanced models like EfficientNet or ResNet.
