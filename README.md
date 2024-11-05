# CS769
# SafeHatQA: Vision-Language Model for Hard Hat Detection

## Cloning our code
To clone our repository run the following command:
```bash
git clone https://github.com/EleannaPanagiotou/CS769-Project.git
```
## Environment setup
To install our environment run the following command:
```bash
conda env create -f environment.yml
```
## To be able to download Hard Hat dataset from kaggle, follow the instructions here: 
1. Run the following command:
    ```bash
    pip install kaggle
    ```
2. https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md : To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials.
3.  Run the following commands:
    ```bash
    mkdir -p ~/.kaggle mv /path/to/kaggle.json ~/.kaggle/ 
    chmod 600 ~/.kaggle/kaggle.json
    ```

## Running DETR model
1. Navigate to the detr_reimplementation folder
    ```bash
    cd detr_reimplementation/
    ```
2. Run the data_preprocessing.py file with the following command:
    ```bash
    python3 data_preprocessing.py
    ```
3. Open the detr_model.ipynb file and run all cells to execute the model

## Running fine-tuned moondream2 
1. Navigate to the moondream2_finetune folder
    ```bash
    cd moondream2_finetune/
    ```
2. Run the data_preprocessing.py file with the following command:
    ```bash
    python3 data_preprocessing.py
    ```
3. Run the following code:
    ```bash
    python3 model.py --mode="train" --model_checkpoint="checkpoints/moondream-ft"
    ```
   After the training process the model will evaluate the fine-tuned moondream2 in train, val and test datasets.

**Note: The codes will be improved and updated until the end of the semester.**




