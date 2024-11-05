# AdvancedNLP

## To install our environment run the following code:
conda env create -f environment.yml

## To be able to download Hard Hat dataset from kaggle, follow the instructions here: 
1. pip install kaggle
2. https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md : To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials.
3.  mkdir -p ~/.kaggle
    mv /path/to/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
4. Run the data_preprocessing.py file 

## Running DETR model
1. Go to detr_reimplementation folder
2. Run the data_preprocessing.py file
3. Open the detr_model.ipynb file and run all cells

## Running fine-tuned moondream2 
1. Go to moondream2_finetune folder
2. Run the data_preprocessing.py file
3. Run the following code:
    python3 model.py --mode="train" --model_checkpoint="checkpoints/moondream-ft"
   After the training process the model will evaluate the fine-tuned moondream2 in train, val and test datasets.

The codes will be improved and updated until the end of the semester.




