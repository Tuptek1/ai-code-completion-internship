import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from sacrebleu import corpus_chrf
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

# Global configuration variables
NUM_SAMPLES = 20            # Number of code samples to generate per file
MIN_MIDDLE_LENGTH = 40       # Minimum length of missing code segment to predict
MAX_MIDDLE_LENGTH = 100      # Maximum length of missing code segment to predict
PREFIX_LIMIT_CHARS = 50      # Max characters in prefix context for the model
SUFFIX_LIMIT_CHARS = 50      # Max characters in suffix context for the model
MAX_NEW_TOKENS = 10          # Limit for generated tokens in completion, affects the length of the completion which means it also affects the accuracy


def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return ""
    except IOError:
        print(f"Error: Unable to read file {file_path}.")
        return ""

def split_code_at_cursor(code, num_examples=1):
    examples = []
    code_length = len(code)

    for _ in range(num_examples):
        cursor_position = random.randint(0, code_length - 1)
        middle_length = random.randint(MIN_MIDDLE_LENGTH, MAX_MIDDLE_LENGTH)
        middle_end = min(cursor_position + middle_length, code_length)

        prefix_start = max(0, cursor_position - PREFIX_LIMIT_CHARS)
        suffix_end = min(middle_end + SUFFIX_LIMIT_CHARS, code_length)

        prefix = code[prefix_start:cursor_position]
        middle = code[cursor_position:middle_end]
        suffix = code[middle_end:suffix_end]

        examples.append({'prefix': prefix, 'middle': middle, 'suffix': suffix})
    
    return examples

def process_directory(directory, num_samples=NUM_SAMPLES):
    examples = []
    
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                code = read_file(file_path)

                if code:  
                    file_examples = split_code_at_cursor(code, num_examples=num_samples)
                    examples.extend(file_examples)
                    if len(examples) >= num_samples:
                        return examples[:num_samples]

    return examples

def generate_completions(model, tokenizer, prefix):
    inputs = tokenizer(prefix, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_metrics(middle, completion):
    exact_match = 1 if middle.strip() in completion.strip() else 0

    chrf_score = corpus_chrf([completion], [[middle]]).score

    rouge = Rouge()
    rouge_scores = rouge.get_scores(completion, middle, avg=True)

    completion_tokens = nltk.word_tokenize(completion)
    middle_tokens = nltk.word_tokenize(middle)
    bleu_score = sentence_bleu([middle_tokens], completion_tokens)

    return exact_match, chrf_score, rouge_scores['rouge-l']['f'], bleu_score

def run_tiny_starcoder_on_examples(example_source_dir):
    examples = process_directory(example_source_dir, num_samples=NUM_SAMPLES)

    try:
        tokenizer = AutoTokenizer.from_pretrained("bigcode/tiny_starcoder_py")
        model = AutoModelForCausalLM.from_pretrained("bigcode/tiny_starcoder_py")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    completions_output_file = "completions_output.txt"
    with open(completions_output_file, 'w') as f_out:
        for i, example in enumerate(examples):
            prefix = example['prefix']
            middle = example['middle']
            suffix = example['suffix']

            generated_completion = generate_completions(model, tokenizer, prefix + middle)
            exact_match, chrf_score, rouge_f_score, bleu_score = calculate_metrics(middle, generated_completion)

            f_out.write(f"Example {i + 1}:\n")
            f_out.write(f"Prefix (Before Cursor):\n{prefix}\n")
            f_out.write(f"Middle (Actual Missing Code):\n{middle}\n")
            f_out.write(f"Tiny Starcoder Generated Completion:\n{generated_completion}\n")
            f_out.write(f"Suffix (After Cursor):\n{suffix}\n\n")
            f_out.write(f"Exact Match: {exact_match}\n")
            f_out.write(f"CHRF Score: {chrf_score:.4f}\n")
            f_out.write(f"ROUGE F-Score: {rouge_f_score:.4f}\n")
            f_out.write(f"BLEU Score: {bleu_score:.4f}\n")
            f_out.write(f"{'=' * 50}\n")

    print(f"Saved completions to {completions_output_file}")

if __name__ == "__main__":
    source_code_directory = "./example_source_codes"  # Adjust to your directory path
    run_tiny_starcoder_on_examples(source_code_directory)
