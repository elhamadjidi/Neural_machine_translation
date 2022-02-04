# Neural Machine Translation (NMT) 
This is a repository for a deep learning model using encoder-decoder architecture to translate german sequence to English automatically.
## prerequisites:
- python 3 
- Keras 2.0 or higher
- NumPy
- matplotlib

All packages used here can be installed by pip as following the example for NumPy:
```
pip install numpy
```
## Dataset:
The code downloads the dataset from https://raw.githubusercontent.com/jbrownlee/Datasets/master/deu.txt, you don't need to download it separately.

## process, train, test
```data_preparation.py``` downloads the dataset and separates German and English and pre-processes the dataset.
```load_data.py``` fits a german and English tokenizer and prepared training, validation, and test data. 
```model.py``` defines the model and trains the model with german and English sequences training data.
```evaluate.py``` evaluates the model on train and test dataset. The BLEU score is the evaluation metric in this project.

## Generate translation
```python3 model.py```
