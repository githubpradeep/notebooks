#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install transformers bitsandbytes accelerate datasets')


# In[5]:


get_ipython().system('pip install peft')


# In[42]:


import re
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset, Dataset

model_name = 'Qwen/Qwen2-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt':tokenizer.apply_chat_template(
            [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ], tokenize = False, add_generation_prompt = True),

        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # for i in range(len(prompts)):
        # if extracted_responses[i] == answer[i]:
        #     print('-'*20, prompts[i])
        #     print("generated ", responses[i])
        #     print("real answer ", answer[i])
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion for completion in completions]
    return [count_xml(c) for c in contents]


# In[2]:


dataset[0]


# In[3]:


import torch
from torch.optim import Adam
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from peft import LoraConfig, get_peft_model
import re
from numbers import Number
from typing import Union
from torch.nn import functional as F
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class REINFORCETrainer:
    def __init__(self, 
                 model_name: str = 'Qwen/Qwen2-1.5B-Instruct',
                 learning_rate: float = 1e-5,
                 kl_coef: float = 0.02,
                 num_epochs: int = 3,
                 batch_size: int = 2,
                 num_samples: int = 5,
                 max_length: int = 1024,
                 device: str = None):
        """
        Initialize the REINFORCE trainer.
        
        Args:
            model_name: Name of the pretrained model
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            num_samples: Number of samples per prompt for RLOO
            max_length: Maximum sequence length
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.kl_coef = kl_coef
        self.num_samples = num_samples
        self.max_length = max_length
        
        # Initialize model and tokenizer with left padding
        logger.info(f"Loading model and tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"  # ✅ Set left padding

        # Fix missing padding token for causal LM
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        DTYPE = torch.float32 if device == "cpu" else torch.bfloat16
        model =  AutoModelForCausalLM.from_pretrained(
                            model_name, trust_remote_code=True,
                            device_map={"": 'cuda'},
                            load_in_4bit= True,
                            torch_dtype=DTYPE,
                        )

        

        config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj"],

            #lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        from peft import prepare_model_for_kbit_training

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)


        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=DTYPE,
        )
        self.ref_model.eval()

        self.model  = get_peft_model(model, config)
        
        
        # Set padding configuration
        # self.tokenizer.padding_side = 'left'  # Fix for decoder-only architecture
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    

    

    def compute_reward(self, generated_text: str, ground_truth: str, ability: str) -> float:
        """Improved reward computation with flexible matching."""
        extracted = self.extract_answer(generated_text)
        # print("generated_text", generated_text)
        # print("extracted", extracted)
        # print("ground_truth", ground_truth)
        gt_processed = self._process_ground_truth(ground_truth)
        
        if ability == 'math':
            rew =  self._numeric_reward(extracted, gt_processed)
            #print("reward", rew)
            return rew
        else:
            return self._semantic_reward(generated_text, gt_processed)

    

    

    def get_logprobs_for_sequence(self, generated_ids: torch.Tensor, input_lengths: torch.Tensor, model=None) -> torch.Tensor:
        """Compute log probabilities for the generated tokens only, considering varying input lengths."""
        model = model if model else self.model
        # Shift input for next-token prediction
        shifted_input = generated_ids[..., :-1].contiguous()
        targets = generated_ids[..., 1:].contiguous()

        # Ensure attention mask is properly created
        attention_mask = (shifted_input != self.tokenizer.pad_token_id).long()

        # Forward pass through the model
        outputs = self.model(shifted_input, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)

        # Gather log probabilities of the actual target tokens
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # Create mask to select only the generated tokens (exclude input)
        mask = torch.zeros_like(targets, dtype=torch.bool, device=targets.device)
        
        # Iterate over batch to set masking dynamically based on `input_lengths`
        for i, input_length in enumerate(input_lengths):
            mask[i, input_length:] = True  # Select only generated tokens

        # Apply mask and sum log probs for generated tokens
        selected_log_probs = target_log_probs * mask
        sequence_log_probs = selected_log_probs.sum(dim=1)

        return sequence_log_probs


    def _train_step(self, batch: Dict[str, List[str]]) -> Tuple[float, List[float]]:
        """Execute a single training step with memory management fixes."""
        
        
        # Prepare inputs
        prompts = batch['prompt']
        ground_truths = batch['answer']

        model = self.model
        tokenizer = self.tokenizer
        
        

        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            #max_length=512,  # Adjust max length if needed
            return_tensors='pt'
        )

        # Move input tensors to the appropriate device
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # Compute actual input lengths (excluding padding)
        input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

        # Ensure pad_token_id is set correctly
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Generate sequences without gradients
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_length,  # Controls new tokens generation
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=self.num_samples,  # Change this if needed
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                use_cache=True  # Enable KV caching for efficiency
            )

        # Extract generated sequences
        sequences = output.sequences
        #print(len(sequences))
        # Tokenize input
        
        # Decode only the generated part of the sequences
        generated_texts = []
        for i, seq in enumerate(sequences):
            new_tokens = seq[input_lengths[i]:]  # Extract only newly generated tokens
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)

        # Print results
        # for i, text in enumerate(generated_texts):
        #     print("**********************************")
        #     print(f"Prompt: {prompts[i]}\nGenerated: {text}\n")

        # Compute rewards
        rewards = []
        r1 = correctness_reward_func(prompts, generated_texts, ground_truths)
        r2 = int_reward_func(generated_texts)
        r3 = strict_format_reward_func(generated_texts)
        r4 = soft_format_reward_func(generated_texts)
        r5 = xmlcount_reward_func(generated_texts)

      
        
        r1 = torch.tensor(r1, device=self.device)
        r2 = torch.tensor(r2, device=self.device)
        r3 = torch.tensor(r1, device=self.device)
        r4 = torch.tensor(r2, device=self.device)
        r5 = torch.tensor(r1, device=self.device)
        rewards = r1+r2+r3+r4+r5
        
        # Normalize rewards
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()
        
        # Compute log probs with gradients (only for generated tokens)
        sequence_log_probs = self.get_logprobs_for_sequence(sequences, input_lengths)


        # with torch.no_grad():
        #     ref_log_probs = self.get_logprobs_for_sequence(sequences, input_lengths, model=self.ref_model)

        # kl_div = F.kl_div(sequence_log_probs, ref_log_probs, reduction='batchmean')
        

        
        # Explicitly delete large tensors and free memory
        del sequences, inputs
        torch.cuda.empty_cache()

        # Compute loss
        #loss = -(sequence_log_probs * rewards).mean()

        loss = -(sequence_log_probs * rewards).mean() #+ self.kl_coef * kl_div  # ✅ REINFORCE Loss with KL penalty
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), rewards.tolist()

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Custom collate function to handle the dataset properly."""
        return {
            'prompt': [item['prompt'] for item in batch],
            'answer': [item['answer'] for item in batch],
        }

    def train(self, dataset: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Train the model using REINFORCE.
        
        Args:
            training_data: List of dictionaries containing structured training data
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Starting training...")
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        metrics = {'epoch_losses': [], 'average_rewards': []}
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            total_rewards = []
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in progress_bar:
                try:
                    loss, rewards = self._train_step(batch)
                    total_loss += loss
                    total_rewards.extend(rewards)
                    
                    progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                except Exception as e:
                    logger.error(f"Error in training step: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            avg_loss = total_loss / len(dataloader)
            avg_reward = np.mean(total_rewards)
            
            metrics['epoch_losses'].append(avg_loss)
            metrics['average_rewards'].append(avg_reward)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}, "
                       f"Average Loss: {avg_loss:.4f}, "
                       f"Average Reward: {avg_reward:.4f}")
        
        return metrics

    def save_model(self, path: str):
        """Save the model and tokenizer."""
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")

# def main():
#     # Example training data
#     training_data = [
#         {
#             'data_source': 'openai/gsm8k',
#             'prompt': [{
#                 'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
#                 'role': 'user'
#             }],
#             'ability': 'math',
#             'reward_model': {
#                 'ground_truth': '72',
#                 'style': 'rule'
#             },
#             'extra_info': {'index': 0, 'split': 'train'}
#         }
#     ]

#     # Initialize trainer
#     trainer = REINFORCETrainer(
#         model_name='gpt2',
#         learning_rate=1e-5,
#         num_epochs=3,
#         batch_size=2,
#         num_samples=5
#     )

#     # Train model
#     try:
#         metrics = trainer.train(train_dataset)
#         trainer.save_model('fine-tuned-gpt2-reinforce')
#         logger.info("Training completed successfully!")
#     except Exception as e:
#         logger.error(f"Training failed: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())

# if __name__ == "__main__":
#     main()


# In[4]:


# import datasets

# num_few_shot = 5
# dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
# dataset = load_dataset(dataset_id, split="train")
# # select a random subset of 50k samples
# dataset = dataset.shuffle(seed=42).select(range(50000))
# train_test_split = dataset.train_test_split(test_size=0.1)
 
# train_dataset = train_test_split["train"]
# test_dataset = train_test_split["test"]
# #test_dataset = dataset['test']
# def extract_solution(solution_str):
#     solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str) # extract the solution after ####
#     assert solution is not None
#     final_solution = solution.group(0)
#     final_solution = final_solution.split('#### ')[1].replace(',', '')
#     return final_solution


# instruction_following = "Let's think step by step and output the final answer after \"####\"."

# # add a row to each data item that represents a unique id
# def make_map_fn(split):

#     def process_fn(example, idx):
#         numbers = example.pop('nums')
#         target = example.pop('target')
#         question = f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."

#         #question = question + ' ' + instruction_following

#         system = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
#         The assistant first thinks about the reasoning process in the mind and then provides the user
#         with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think>.
#         '''

#         question =f"""
#         <|im_start|> {system}
#         <|im_end|>
#         <|im_start|> user
#             {question}
#         <|im_end|>
#         <|im_start|>  
#         assistant  <think>"""


#         # question =f"""
#         # <|im_start|>You are a helpful AI assistant. 
#         # "Let's think step by step and output the final answer after \"####\"."  <|im_end|>
#         # <|im_start|> user
#         #     {question}
#         # <|im_end|>
#         # <|im_start|>  
#         # assistant"""


        

#         # answer = example.pop('answer')
#         # solution = extract_solution(answer)
#         data = {
#             "data_source": dataset_id,
#             "prompt": [{
#                 "role": "user",
#                 "content": question
#             }],
#             "ability": "math",
#             "reward_model": {
#                 "style": "rule",
#                 "ground_truth": target
#             },
#             "extra_info": {
#                 'split': split,
#                 'index': idx
#             },
#             "nums": numbers
#         }
#         return data

#     return process_fn

# import re
# train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
# test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)


# In[5]:


def generate_response(model, tokenizer, prompt, device, max_length=200):
    """Generate a response for a given prompt."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# In[6]:


training_data = [
    {
        'data_source': 'openai/gsm8k',
        'prompt': [{
            'content': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
            'role': 'user'
        }],
        'ability': 'math',
        'reward_model': {
            'ground_truth': '72',
            'style': 'rule'
        },
        'extra_info': {'index': 0, 'split': 'train'}
    }
]

# Initialize trainer
trainer = REINFORCETrainer(
    model_name='Qwen/Qwen2-1.5B-Instruct',
    learning_rate=1e-5,
    num_epochs=3,
    batch_size=2,
    num_samples=1
)



# In[57]:


trainer.batch_size = 16
trainer.max_length = 512
dataset = dataset.shuffle()
# After loading your model:
trainer.model.gradient_checkpointing_enable()


# In[61]:


# Train model
try:
    metrics = trainer.train(dataset)#(training_data)
    #trainer.save_model('fine-tuned-gpt2-reinforce')
    logger.info("Training completed successfully!")
except Exception as e:
    logger.error(f"Training failed: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())


# In[62]:


trainer.model.push_to_hub("Neuranest/Qwen2-1.5B-Instruct-Math", use_temp_dir=True, safe_serialization=True,
    token='')
# trainer.tokenizer.push_to_hub("Neuranest/Qwen2-1.5B-Instruct-Math", use_temp_dir=True,
#     token='')


# In[ ]:





# In[37]:


get_ipython().system('pip install lm-eval')


# In[39]:


get_ipython().system('lm_eval --model hf      --model_args pretrained=Qwen/Qwen2-1.5B-Instruct      --tasks gsm8k_cot      --device cuda:0      --batch_size 8      --apply_chat_template      --fewshot_as_multiturn')


# In[63]:


get_ipython().system('lm_eval --model hf      --model_args pretrained=Qwen/Qwen2-1.5B-Instruct,peft=Neuranest/Qwen2-1.5B-Instruct-Math      --tasks gsm8k_cot      --device cuda:0      --batch_size 8      --apply_chat_template      --fewshot_as_multiturn')


# In[66]:


get_ipython().system('lm_eval --model hf      --model_args pretrained=Qwen/Qwen2-1.5B-Instruct      --tasks gsm8k_cot_zeroshot      --device cuda:0      --batch_size 16      --apply_chat_template')


# In[64]:


get_ipython().system('lm_eval --model hf      --model_args pretrained=Qwen/Qwen2-1.5B-Instruct,peft=Neuranest/Qwen2-1.5B-Instruct-Math      --tasks gsm8k_cot_zeroshot      --device cuda:0      --batch_size 8      --apply_chat_template')


# In[1]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
model_name = 'Qwen/Qwen2-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = dataset
# Example prompts
prompts = [train_dataset[3]['prompt'], train_dataset[1]['prompt']]

# Tokenize the input prompts
inputs = tokenizer(
    prompts,
    padding=True,
    truncation=True,
    max_length=512,  # Adjust max length if needed
    return_tensors='pt'
)

# Move input tensors to the appropriate device
inputs = {key: value.to(trainer.model.device) for key, value in inputs.items()}

# Compute actual input lengths (excluding padding)
input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

# Ensure pad_token_id is set correctly
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

# Generate sequences without gradients
with torch.no_grad():
    output = trainer.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5000,  # Controls new tokens generation
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1,  # Change this if needed
        pad_token_id=pad_token_id,
        return_dict_in_generate=True,
        use_cache=True  # Enable KV caching for efficiency
    )

# Extract generated sequences
sequences = output.sequences

# Extract only the generated portion correctly for variable-length sequences
generated_texts = []
for i, seq in enumerate(sequences):
    new_tokens = seq[input_lengths[i]:]  # Extract only newly generated tokens
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    generated_texts.append(generated_text)

# Print results
for i, text in enumerate(generated_texts):
    print("**********************************")
    print(f"Prompt: {prompts[i]}\nGenerated: {text}\n")


# In[ ]:





# In[23]:


train_dataset[3]


# In[9]:


get_ipython().system('pip install datasets')


# In[7]:





# In[10]:





# In[12]:


train_dataset[10]['prompt'][0]['content']


# In[13]:


def extract_answer(text: str) -> str:
        """Extract numerical answer after ####."""
        try:
            parts = text.split('####')
            if len(parts) > 1:
                answer = parts[1].strip()
                import re
                numbers = re.findall(r'\d+', answer)
                return numbers[0] if numbers else ''
            return ''
        except Exception:
            return ''


# In[14]:


inst= """
        <|im_start|>You are a helpful AI assistant.
        Let's think step by step and output the final answer after \"####\"."
         <|im_end|>
        <|im_start|> user
        X = 2*3. What is X
        
<|im_end|>
<|im_start|>   
assistant"""

generate_response(trainer.model, trainer.tokenizer, inst, 'cuda', max_length=100)


# In[ ]:


extract_answer("""
'\n        You are a helpful AI assistant. \n         user\n        X = 2*3. What is X\n\n  "Let\'s think step by step and output the final answer after "####"."  \nassistant\nX = 2*3\n\nX = 6\n\nTo find the value of X, we multiply 2 by 2:\n\n3 * 3 = 6\n4 * 4 = 6\n\nTherefore, X = 6. The answer is X = 6.""")


# In[ ]:




