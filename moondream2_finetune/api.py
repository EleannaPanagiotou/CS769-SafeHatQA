from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch


#######Pretrained Model#######

model_pre = "vikhyatk/moondream2"
revision = "2024-08-26"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    model_pre, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_pre, revision=revision)

# image = Image.open('processed_splits/test/hard_hat_workers1.png')
# image = Image.open('processed_splits/train/hard_hat_workers4907.png')
# image = Image.open('processed_splits/test/hard_hat_workers3447.png')
# image = Image.open('processed_splits/train/hard_hat_workers74.png')
# # image = Image.open('final_processed_splits/test/hard_hat_workers2277.png')
# image = Image.open('processed_splits/test/hard_hat_workers187.png')
# image = Image.open('processed_splits/test/hard_hat_workers4626.png')
# image = Image.open('processed_splits/test/hard_hat_workers2308.png')
# image = Image.open('processed_splits/test/hard_hat_workers2952.png')
image = Image.open('processed_splits/val/hard_hat_workers1836.png')
enc_image = model.encode_image(image)


questions = [
    "How many people are in this image?",
    "How many hard hats are in this image?",
    "How many people without hard hats are in this image?",
    "Are there any people in this image?",
    "Are all people wearing hard hats in this image?",
    "Are there any people without hard hats in this image?",
    "Are there more people with hard hats than without?",
    "Are there more people without hard hats than with?"
]

for question in questions:
    answer = model.answer_question(enc_image, question, tokenizer)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 30)


#######Finetuned Model#######

model_ft = "checkpoints_1/moondream-ft"

model = AutoModelForCausalLM.from_pretrained(
    model_ft, trust_remote_code=True, revision=revision, attn_implementation=None, 
            torch_dtype=DTYPE, 
            device_map={"": DEVICE}
)

# image = Image.open('processed_splits/train/hard_hat_workers4907.png')
# image = Image.open('processed_splits/test/hard_hat_workers1.png')
# image = Image.open('processed_splits/test/hard_hat_workers3447.png')
# image = Image.open('processed_splits/test/hard_hat_workers187.png')
# image = Image.open('processed_splits/test/hard_hat_workers4626.png')
# image = Image.open('processed_splits/train/hard_hat_workers74.png')
# image = Image.open('final_processed_splits/test/hard_hat_workers2277.png')
# image = Image.open('processed_splits/test/hard_hat_workers2308.png')
# image = Image.open('processed_splits/test/hard_hat_workers2952.png')
image = Image.open('processed_splits/val/hard_hat_workers1836.png')
enc_image = model.encode_image(image)


questions = [
    "How many people are in this image?",
    "How many hard hats are in this image?",
    "How many people without hard hats are in this image?",
    "Are there any people in this image?",
    "Are all people wearing hard hats in this image?",
    "Are there any people without hard hats in this image?",
    "Are there more people with hard hats than without?",
    "Are there more people without hard hats than with?"
]

for question in questions:
    answer = model.answer_question(enc_image, question, tokenizer)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 30)