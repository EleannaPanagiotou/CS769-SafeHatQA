import json
import os
import math
import torch
import random
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from word2number import w2n
import argparse
from torch.utils.data import Dataset, DataLoader
from bitsandbytes.optim import Adam8bit
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from PIL import Image

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Argument parser for training or evaluation mode
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "evaluate"], default="evaluate", help="Mode to run the script in")
parser.add_argument("--model_checkpoint", type=str, help="Path to a saved model checkpoint for evaluation")
args = parser.parse_args()

# Model and training configurations
EPOCHS = 20
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
LR = 1e-5
USE_WANDB = False  # Set to True if you want to log with Weights and Biases
ANSWER_EOS = "<|endoftext|>"
IMG_TOKENS = 729
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
MD_REVISION = "2024-07-23"

# Dataset class for Hard Hat Q&A dataset
class HardHatQADataset(Dataset):
    def __init__(self, json_file):
        # Load the JSON data with image paths and QA pairs
        with open(json_file, "r") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return {
            "image": image,
            "qa": sample["qa"]
        }

# Load datasets
datasets = {
    "train": HardHatQADataset("processed_splits/train.json"),
    "val": HardHatQADataset("processed_splits/val.json"),
    "test": HardHatQADataset("processed_splits/test.json"),
}

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
    attn_implementation=None,  # Disable FlashAttention2 if not available
    torch_dtype=DTYPE, device_map={"": DEVICE}
)

# Collate function for DataLoader
def collate_fn(batch):
    images = [sample['image'] for sample in batch]
    images = [moondream.vision_encoder.preprocess(image) for image in images]

    labels_acc = []
    tokens_acc = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        for qa in sample['qa']:
            q_t = tokenizer(f"\n\nQuestion: {qa['question']}\n\nAnswer:", add_special_tokens=False).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))

            a_t = tokenizer(f" {qa['answer']}{ANSWER_EOS}", add_special_tokens=False).input_ids
            toks.extend(a_t)
            labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)

    max_len = max(len(labels) for labels in labels_acc)
    attn_mask_acc = []

    for i in range(len(batch)):
        len_i = len(labels_acc[i])
        pad_i = max_len - len_i

        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    return (
        images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )

# Learning rate scheduler
def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

# Loss computation
def compute_loss(batch):
    images, tokens, labels, attn_mask = batch
    tokens, labels, attn_mask = tokens.to(DEVICE), labels.to(DEVICE), attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = moondream.vision_encoder(images)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )
    return outputs.loss

# Function to extract the first number from a text string
def extract_number(text):
    try:
        # Try converting words to numbers using word2number if digits aren't found
        match = re.search(r'\d+', text)  # Check for a numeric string
        if match:
            return int(match.group())
        else:
            # If no digit found, try converting words to numbers
            return w2n.word_to_num(text.lower())
    except ValueError:
        # Return None if conversion fails
        return None

# Regression-based evaluation function
def evaluate_model_regression(model, dataset):
    predictions = []
    ground_truths = []

    for sample in tqdm(dataset):
        encoded_image = model.encode_image(sample['image'])
        question = sample['qa'][1]['question']
        ground_truth = extract_number(sample['qa'][1]['answer'])
        
        # Generate answer and extract number
        md_answer = model.answer_question(
            encoded_image,
            question,
            tokenizer=tokenizer,
            num_beams=4,
            no_repeat_ngram_size=5,
            early_stopping=True
        ).strip()
        predicted_number = extract_number(md_answer)

        # Append extracted numbers to lists for metric calculations
        if ground_truth is not None and predicted_number is not None:
            ground_truths.append(ground_truth)
            predictions.append(predicted_number)

    # Calculate MAE, MSE, RMSE
    mae = mean_absolute_error(ground_truths, predictions)
    mse = mean_squared_error(ground_truths, predictions)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Evaluation function with regression metrics
def evaluate_all_splits(model, datasets):
    for split_name, dataset in datasets.items():
        mae, mse, rmse = evaluate_model_regression(model, dataset)
        print(f"{split_name.capitalize()} set - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# Training code
if args.mode == "train":
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Set up optimizer and scheduler
    total_steps = EPOCHS * len(train_loader) // GRAD_ACCUM_STEPS
    optimizer = Adam8bit([{"params": moondream.text_model.parameters()}], lr=LR * 0.1, betas=(0.9, 0.95), eps=1e-5)

    # Training loop
    step = 0
    for epoch in range(EPOCHS):
        moondream.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            step += 1
            loss = compute_loss(batch)
            loss.backward()

            if step % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(step // GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")
        if epoch % 3 == 0:
            moondream.eval()
            print("Evaluating on the validation set...")
            mae, mse, rmse = evaluate_model_regression(moondream, datasets["val"])
            print(f"Validation set - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")


    # Save model checkpoint after training
    moondream.save_pretrained("checkpoints/moondream-ft")

    moondream.eval()
    print("Evaluating on the train set...")
    mae, mse, rmse = evaluate_model_regression(moondream, datasets["train"])
    print(f"Train set - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

    print("Evaluating on the validation set...")
    mae, mse, rmse = evaluate_model_regression(moondream, datasets["val"])
    print(f"Validation set - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

    print("Evaluating on the test set...")
    mae, mse, rmse = evaluate_model_regression(moondream, datasets["test"])
    print(f"Test set - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# Evaluation code
elif args.mode == "evaluate":
    moondream.eval()
    # Run evaluation on train, validation, and test sets
    evaluate_all_splits(moondream, {"train": datasets["train"], "val": datasets["val"], "test": datasets["test"]})