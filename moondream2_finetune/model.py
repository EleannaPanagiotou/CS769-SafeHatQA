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
from torch.optim.lr_scheduler import LambdaLR

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
EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
LR = 1e-5
USE_WANDB = False 
ANSWER_EOS = "<|endoftext|>"
IMG_TOKENS = 729
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
MD_REVISION = "2024-07-23"

# Dataset class for Hard Hat Q&A dataset
class HardHatQADataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return {
            "image": image,
            "qa": sample["qa"]  # Include question, answer, and type
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
    attn_implementation=None,  
    torch_dtype=DTYPE, device_map={"": DEVICE}
)

# Collate function for DataLoader
def collate_fn(batch):
    images = [sample['image'] for sample in batch]
    images = [moondream.vision_encoder.preprocess(image) for image in images]

    tokens_acc = []
    labels_acc = []
    types_acc = []  # New accumulator for answer types

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)
        types = []

        for qa in sample['qa']:
            # Tokenize the question
            q_t = tokenizer(f"\n\nQuestion: {qa['question']}\n\nAnswer:", add_special_tokens=False).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))

            # Tokenize the answer
            answer = qa['answer']
            answer_type = qa.get('type', 'numerical')  # Default to numerical if type is missing
            if answer_type == 'yes_no':
                a_t = [1] if answer.lower() == "yes" else [0]  # Binary labels for Yes/No
                types.append('yes_no')
            else:
                a_t = tokenizer(f" {answer}{ANSWER_EOS}", add_special_tokens=False).input_ids
                types.append('numerical')

            toks.extend(a_t)
            labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)
        types_acc.append(types)

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
        types_acc,  # Pass answer types
    )

# Learning rate scheduler
def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

def compute_loss(batch):
    images, tokens, labels, attn_mask, types = batch
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
        match = re.search(r'\d+', text) 
        if match:
            return int(match.group())
        else:
            # If no digit found, try converting words to numbers
            return w2n.word_to_num(text.lower())
    except ValueError:
        return None
    
def extract_yes_no(answer):
    """
    Extract 'Yes' or 'No' from the given answer in a case-insensitive manner.

    Args:
        answer (str): The model's generated answer.

    Returns:
        str: 'Yes' or 'No' if found, otherwise 'No prediction'.
    """
    match = re.search(r"\b(yes|no|not)\b", answer, re.IGNORECASE)
    if match:
        extracted = match.group(0).lower()
        return "No" if extracted == "not" else extracted.capitalize()
    
    return match.group(0).capitalize() if match else "No prediction"

question_groups = {
    "count": ["How many people are in this image?", 
              "How many hard hats are in this image?", 
              "How many people without hard hats are in this image?"],
    "comparison": ["Are there more people with hard hats than without?", 
                   "Are there more people without hard hats than with?"],
    "yes_no": ["Are there any people in this image?",
               "Are all people wearing hard hats in this image?",
               "Are there any people without hard hats in this image?"]
}

def collate_fn_eval(batch):
    """
    Collate function for evaluation.
    Only processes images and QA pairs.
    """
    images = [sample['image'] for sample in batch]
    images = [moondream.vision_encoder.preprocess(image) for image in images]  # Preprocess images
    qa_pairs = [sample['qa'] for sample in batch]  # Extract QA pairs

    return images, qa_pairs

def evaluate_model(model, dataset, question_groups, tokenizer):
    question_metrics = {q: {"correct": 0, "total": 0} for q in question_groups["count"] + question_groups["comparison"] + question_groups["yes_no"]}
    group_metrics = {group: {"correct": 0, "total": 0} for group in question_groups.keys()}

    numerical_predictions = []
    numerical_ground_truths = []

    model.eval() 
    with torch.no_grad(): 
        print(type(dataset))
        for sample in tqdm(dataset):
            image = sample['image']
            qa_pairs = sample['qa']
            encoded_image = model.encode_image(image)

            for qa in qa_pairs:
                question = qa['question']
                answer_type = qa['type']
                ground_truth = qa['answer']

                # Generate model answer
                model_answer = model.answer_question(
                    encoded_image,
                    question,
                    tokenizer=tokenizer,
                    num_beams=4,
                    no_repeat_ngram_size=5,
                    early_stopping=True
                ).strip()

                # Evaluate Yes/No questions
                if answer_type == "yes_no":
                    extracted_answer = extract_yes_no(model_answer)
                    is_correct = extracted_answer.lower() == ground_truth.lower()
                    question_metrics[question]["correct"] += int(is_correct)
                    question_metrics[question]["total"] += 1
                    group_metrics["yes_no"]["correct"] += int(is_correct)
                    group_metrics["yes_no"]["total"] += 1

                # Evaluate Count questions
                elif answer_type == "count":
                    ground_truth_num = extract_number(ground_truth)
                    predicted_num = extract_number(model_answer)
                    if ground_truth_num is not None and predicted_num is not None:
                        numerical_ground_truths.append(ground_truth_num)
                        numerical_predictions.append(predicted_num)

                        # Count as correct if numbers match
                        is_correct = ground_truth_num == predicted_num
                        question_metrics[question]["correct"] += int(is_correct)
                        question_metrics[question]["total"] += 1
                        group_metrics["count"]["correct"] += int(is_correct)
                        group_metrics["count"]["total"] += 1

                # Evaluate Comparison questions
                if question in question_groups["comparison"]:
                    extracted_answer = extract_yes_no(model_answer)
                    is_correct = extracted_answer.lower() == ground_truth.lower()
                    question_metrics[question]["correct"] += int(is_correct)
                    question_metrics[question]["total"] += 1
                    group_metrics["comparison"]["correct"] += int(is_correct)
                    group_metrics["comparison"]["total"] += 1

    # Calculate overall metrics for numerical answers
    mae = mean_absolute_error(numerical_ground_truths, numerical_predictions) if numerical_predictions else float('nan')
    mse = mean_squared_error(numerical_ground_truths, numerical_predictions) if numerical_predictions else float('nan')
    rmse = np.sqrt(mse) if numerical_predictions else float('nan')

    # Calculate accuracy for each question and group
    question_accuracies = {q: m["correct"] / m["total"] if m["total"] > 0 else 0 for q, m in question_metrics.items()}
    group_accuracies = {g: m["correct"] / m["total"] if m["total"] > 0 else 0 for g, m in group_metrics.items()}

    # Display individual question accuracies
    print("Question-level Metrics:")
    for question, accuracy in question_accuracies.items():
        print(f" - {question}: {accuracy:.2f}")

    # Display group-level accuracies
    print("\nGroup-level Metrics:")
    for group, accuracy in group_accuracies.items():
        print(f" - {group.capitalize()} Accuracy: {accuracy:.2f}")

    # Display numerical metrics
    print("\nNumerical Metrics:")
    print(f" - MAE: {mae:.2f}")
    print(f" - MSE: {mse:.2f}")
    print(f" - RMSE: {rmse:.2f}")


if args.mode == "train":
    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    optimizer = Adam8bit([{"params": moondream.parameters()}], lr=LR * 0.1, betas=(0.9, 0.95), eps=1e-5)

    # Set up scheduler
    total_steps = EPOCHS * len(train_loader) // GRAD_ACCUM_STEPS

    # Training loop
    step = 0
    for epoch in range(EPOCHS):
        moondream.train()
        epoch_loss = 0  

        for batch in tqdm(train_loader):
            images, tokens, labels, attn_mask, types = batch

            # Ensure all tensors and model are on the correct device
            tokens, labels, attn_mask = tokens.to(DEVICE), labels.to(DEVICE), attn_mask.to(DEVICE)
            moondream = moondream.to(DEVICE)

            # Compute loss
            loss = compute_loss(batch)
            epoch_loss += loss.item()

            if not loss.requires_grad:
                print("Debugging Loss:")
                print(f"Loss: {loss}, Requires Grad: {loss.requires_grad}")
                raise RuntimeError("Loss does not require grad. Check model or loss computation.")

            loss.backward()

            # Gradient accumulation
            if step % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                lr = lr_schedule(step // GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr  
                optimizer.zero_grad()

            step += 1

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss / len(train_loader):.4f}")

        # Save model checkpoint after training
        savepath = "checkpoints_" + str(epoch) + "/moondream-ft"
        moondream.save_pretrained(savepath)

# Evaluation-only mode
elif args.mode == "evaluate":
    # Load model checkpoint
    if not args.model_checkpoint:
        moondream = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", 
            revision=MD_REVISION, 
            trust_remote_code=True,
            attn_implementation=None,  
            torch_dtype=DTYPE, 
            device_map={"": DEVICE}
        )
    else:
        moondream = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint, 
            torch_dtype=DTYPE, 
            device_map={"": DEVICE}, 
            trust_remote_code=True
        )
    
    moondream.eval()

    # Iterate through datasets for evaluation
    for split_name, dataset in datasets.items():
        print(f"Evaluating on the {split_name} split...")
        evaluate_model(moondream, dataset, question_groups, tokenizer)