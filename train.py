from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config, latest_weights_file_path

# Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchmetrics 

import sentencepiece as spm
from pathlib import Path

from datasets import load_dataset

from tqdm import tqdm

import os
import warnings

def beam_search(model, source, source_mask, tokenizer_tgt, max_len, device, beam_size=5):
    eos_idx = tokenizer_tgt.eos_id()
    pad_idx = tokenizer_tgt.pad_id()

    encoder_output = model.encode(source, source_mask)
    
    # Initialize with a single pad token
    beams = [(torch.full((1, 1), pad_idx, device=device), 0.0)]
    completed_beams = []

    for _ in range(max_len):
        new_beams = []
        for decoder_input, score in beams:
            if decoder_input.size(1) > 1 and decoder_input[0, -1].item() == eos_idx:
                completed_beams.append((decoder_input, score))
                continue

            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            
            top_scores, top_tokens = torch.topk(prob, beam_size, dim=1)
            
            for token, token_score in zip(top_tokens[0], top_scores[0]):
                new_decoder_input = torch.cat([decoder_input, token.unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score + token_score.item()
                new_beams.append((new_decoder_input, new_score))

        # Keep only the top-k beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Early stopping if all beams have completed
        if len(completed_beams) == beam_size:
            break

    # If no beam has completed, add all current beams to completed_beams
    if not completed_beams:
        completed_beams = beams

    best_beam = max(completed_beams, key=lambda x: x[1])
    return best_beam[0][:, 1:]  # Remove the initial pad token

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=10):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = beam_search(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)
    
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            
            
            model_out_tokens = model_out.detach().cpu().numpy().tolist()
            model_out_tokens = [token for token in model_out_tokens if token != tokenizer_tgt.pad_id()]
            model_out_text = ''.join(tokenizer_tgt.decode(model_out_tokens))

            if not model_out_text or model_out_text.isspace():
                model_out_text = tokenizer_tgt.decode([model_out_tokens[0]]) + tokenizer_tgt.decode([tokenizer_tgt.eos_id()])
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, [[text] for text in expected])
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_or_build_tokenizer(config, lang):
    tokenizer_path = Path(f"tokenizer_{lang}.model")
    if not tokenizer_path.exists():
        # Create text file with all sentences for training the tokenizer
        with open(f"{lang}_sentences.txt", "w", encoding="utf-8") as f:
            for item in load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train'):
                f.write(item['translation'][lang] + "\n")
        
        # Train tokenizer
        spm.SentencePieceTrainer.train(
            input=f"{lang}_sentences.txt",
            model_prefix=f"tokenizer_{lang}",
            vocab_size=32000,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            eos_id=2,
            bos_id=3,  # Need BOS for some reason...
            pad_piece="[PAD]",
            unk_piece="[UNK]",
            eos_piece="[EOS]",
            bos_piece="[BOS]", 
            user_defined_symbols="[SEP],[CLS],[MASK]",
        )
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    return tokenizer

def get_all_sentences(ds, lang):

    for item in ds:
        yield item['translation'][lang]

def get_ds(config):

    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Remove incorrect translation pairs
    correct_indices = list(range(0, 20000)) + list(range(24000, 60000)) + list(range(85000, len(ds_raw)))
    ds_raw = ds_raw.select(correct_indices)

    tokenizer_src = get_or_build_tokenizer(config, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, config['lang_tgt'])

    # Filter out sequences that are too long
    def filter_long_sequences(example):
        src_ids = tokenizer_src.encode(example['translation'][config['lang_src']])
        tgt_ids = tokenizer_tgt.encode(example['translation'][config['lang_tgt']])
        return len(src_ids) <= config['seq_len'] and len(tgt_ids) <= config['seq_len']

    ds_raw = ds_raw.filter(filter_long_sequences)

    # train, validate data partitioning split

    train_ds_size = int(0.8 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']])
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], num_workers = 4, pin_memory = True, shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, num_workers = 2, pin_memory = True, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, tokenizer_src, tokenizer_tgt):
    src_vocab_size = tokenizer_src.get_piece_size()  
    tgt_vocab_size = tokenizer_tgt.get_piece_size()  
    
    model = build_transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        config["seq_len"], 
        config['seq_len'], 
        d_model=config['d_model']
    )

    return model

def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    device = torch.device(device)

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    tgt_vocab_size = tokenizer_tgt.get_piece_size()
    
    model = get_model(config, tokenizer_src, tokenizer_tgt).to(device)
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    
    if config['preload']:
        model_filename = latest_weights_file_path(config) if config['preload'] == 'latest' else get_weights_file_path(config, config['preload'])
        if model_filename and Path(model_filename).exists():
            print(f'Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
        else:
            print(f'No model found at {model_filename}. Starting from scratch.')

    pad_token_id = tokenizer_src.pad_id()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)

    # Gradient Accumulation
    accumulation_steps = 4

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            
            # Compute loss
            loss = loss_fn(output.view(-1, tgt_vocab_size), label.view(-1))
            
            # Normalize the loss to account for accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()

            # Update weights every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_iterator.set_postfix({"loss": f"{loss.item() * accumulation_steps:6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item() * accumulation_steps, global_step)
            writer.flush()

            global_step += 1

        # Run validation at end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Save model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

            