import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('..')

from time import perf_counter

import torch
import torch.nn.functional as F
import torch.nn as nn
from mamba_lm import from_pretrained
from mamba_lm import MambaLM, MambaLMConfig

from transformers import AutoTokenizer

import datasets

import numpy as np
import random


# Automated device selection based on available backends
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available() and False
        else "cpu"
    )

print(f"> Using {device} device")

def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f"{path}/{f}")
    return files

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_checkpoint(filepath, model, scheduler, optimizer):
    print(f"> Loading model from: {filepath}")
    try:
        loaded_checkpoint = torch.load(filepath, map_location=device)

        loaded_epoch = loaded_checkpoint['epoch']
        loaded_model = model
        loaded_scheduler = scheduler
        loaded_optimizer = optimizer

        loaded_model.load_state_dict(loaded_checkpoint['model_state'])
        if scheduler is not None:
            loaded_scheduler.load_state_dict(loaded_checkpoint['scheduler_state'])
        if optimizer is not None:
            loaded_optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
        
        print("> Loaded model")
        return True, loaded_epoch, loaded_model, loaded_scheduler, loaded_optimizer
    except Exception as e:
        print("> Cannot load model")
        return False, 0, model, scheduler, optimizer

def train(pretrained=False):
    # Training parameters
    '''
    epochs - number of epochs during training
    batch_size - size of a single batch during training
    seq_length - number of tokens in model's context during training
    learning_rate - initial learning rate of the training
    model_path - path to the saved weights; if empty it'll save there new weights during training
    '''
    epochs = 150
    batch_size = 64 #32 for 24GB and 130m model
    seq_length = 128
    learning_rate = 1e-3
    model_path = f'saves/model.pth'

    # Usage of datasets' built in datasets
    #dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')
    dataset = datasets.load_dataset('text', data_files={'train': listdir_nohidden("./gutenberg")}, encoding='utf-8',encoding_errors='ignore')

    # Usage of custom txt datasets
    '''
    In order to load custom training data add filepaths to the list
    For example to use one txt file change the name of the file in the command below:

    dataset = datasets.load_dataset('text', data_files={'train': ['austen-emma.txt']})

    For more files add them to the list after comma

    https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html
    '''

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    # Add eos tokens
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    # Map tokenizer to the dataset
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer.tokenize(example['text'], truncation=True)} 
    tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], 
        fn_kwargs={'tokenizer': tokenizer})
    
    # Prepare and load tokenizer's vocabulary for later use
    vocab = tokenizer.vocab
    print(f"Vocab size: {len(vocab)}")

    
    # Select the wanted model
    '''
    If pretrained==True - the script loads pretrained mamba weights specified by the string.
    If pretrained==False - the script creates a new MambaLM model with parameters specified in config variable
    '''
    if pretrained:
        model = from_pretrained('state-spaces/mamba-130m').to(device)
    else:
        config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(tokenizer.vocab))
        model = MambaLM(config).to(device)

    # Create optimizer and pass the model
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                            optim,
                                                            mode='min',
                                                            factor=0.1, #factor by which the lr is multiplied
                                                            patience=2,
                                                        )

    # Load previously trained weights
    ''' 
    If the model is the same it will load previous weights located in specified path
    If the model differs or the path is empty it'll skip loading and train from scratch
    '''
    _, epoch, model, scheduler, optim = load_checkpoint(model_path, model, scheduler, optim)

    
    # Create data loader functions
    def get_data(dataset, vocab, batch_size):
        data = []                                   
        for example in dataset:
            if example['tokens']:
                tokens = [vocab[token] for token in example['tokens']]
                data.extend(tokens)
        
        data = torch.LongTensor(data)              
        num_batches = data.shape[0] // batch_size 
        data = data[:num_batches * batch_size]                       
        data = data.view(batch_size, num_batches)
        return data     

    def get_batch(data, seq_len, idx):
        src = data[:, idx:idx+seq_len]
        target = data[:, idx+1:idx+seq_len+1]
        return src, target


    # Get data and apply tokenizer to the dataset
    train_data = get_data(tokenized_dataset['train'], vocab, batch_size)
    print(f"Train data length before: {train_data.shape[-1]}")
    

    # Training loop
    t0_start = perf_counter()
    for z in range(epoch, epochs):
        idx = 0
        avg_loss = 0
        print(f"\n> Epoch {z+1}/{epochs}")

        t2_start = perf_counter()
        for i in range(train_data.shape[-1]):   
            model.train()
            t1_start = perf_counter()

            input, output = get_batch(train_data, seq_length, idx)
            output = output.reshape(-1)
            input = input.to(device)
            output = output.to(device)

            logits = model(input)

            # If the batch is not complete - skip
            if (logits.view(-1, logits.size(-1)).shape[0] != output.view(-1).shape[0]):
                print("skip")
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output)
                avg_loss += loss.item()

                optim.zero_grad()
                loss.backward()
                optim.step()

                t1_stop = perf_counter()

                # Print the progress during training and save the model
                if i%10==0:
                    print(f"\r> Batch: {idx}/{train_data.shape[-1]-seq_length} loss: {avg_loss/(i+1):.5f} time: {t1_stop-t1_start:.2f} sec ", end="")

                    checkpoint = {
                        'epoch': z,
                        'model_state': model.state_dict(),
                        'optimizer_state': optim.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                    }
                    torch.save(checkpoint, model_path)

            # Increment idx
            idx += 1
            if idx >= train_data.shape[-1] - seq_length:
                idx = 0
                break

        t2_stop = perf_counter()
        print(f"\n> Epoch time: {t2_stop - t2_start:.3f} seconds")
        # Update schedulers
        scheduler.step(avg_loss/(i+1))

    t0_stop = perf_counter()
    print(f"\n> Finished training in: {t0_stop-t0_start} seconds")

    print("> Generating answer: ")
    # Generate sample text after training
    output = model.generate(tokenizer, "She was the youngest of the two daughters of a most affectionate "
                            , num_tokens=50
                            , temperature=1.0
                            , top_k=None)

    print(f"Answer: {output}")


# Sample generation based on trained model
def my_gen(pretrained=False):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    if pretrained:
        model = from_pretrained('state-spaces/mamba-130m').to(device)
    else:
        config = MambaLMConfig(d_model=16, n_layers=4, vocab_size=len(tokenizer.vocab))
        model = MambaLM(config).to(device)

    # Load weights
    isLoaded, _, model, *_ = load_checkpoint(f'saves/model.pth', model, None, None)
    if (not isLoaded):
        return

    # Generate text based on prompt
    output = model.generate(tokenizer, "She was the youngest of the two daughters "
                            , num_tokens=50
                            , temperature=0.8
                            , top_k=None)

    print(f"Answer: {output}")

def prepare_folders():
    try:
        os.makedirs("./saves/")
    except:
        pass

if __name__ == "__main__":
    seed_everything(534)
    prepare_folders()

    train(pretrained=False)
    #my_gen(pretrained=False)