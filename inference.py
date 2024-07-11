import torch
import sentencepiece as spm

from model import build_transformer
from config import get_config, latest_weights_file_path
from dataset import causal_mask

def translate(phrase, model, tokenizer_src, tokenizer_tgt, device, max_len=512):
    model.eval()
    
    source = tokenizer_src.encode(phrase)
    source = torch.tensor(source).unsqueeze(0).to(device)  # Add batch dimension
    source_mask = (source != tokenizer_src.pad_id()).unsqueeze(0).unsqueeze(1).int().to(device)
    
    # Initialize target with pad token
    target = torch.ones(1, 1).fill_(tokenizer_tgt.pad_id()).type_as(source).to(device)
    
    for i in range(max_len - 1):
        target_mask = causal_mask(target.size(1)).type_as(source_mask).to(device)
        
        # Get the model output
        with torch.no_grad():
            output = model(source, target, source_mask, target_mask)
        
        # Get the next word prediction
        prob = output[:, -1]
        _, next_word = torch.max(prob, dim=1)
        
        # Add the next word to the target sequence
        target = torch.cat([target, torch.ones(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == tokenizer_tgt.eos_id():
            break
    
    translated_sentence = tokenizer_tgt.decode(target.squeeze().tolist())
    
    return translated_sentence

def load_model(config):
  
    tokenizer_src = spm.SentencePieceProcessor()
    tokenizer_src.load(f"tokenizer_{config['lang_src']}.model")
    tokenizer_tgt = spm.SentencePieceProcessor()
    tokenizer_tgt.load(f"tokenizer_{config['lang_tgt']}.model")
    
    model = build_transformer(tokenizer_src.get_piece_size(), tokenizer_tgt.get_piece_size(), 
                              config['seq_len'], config['seq_len'], d_model=config['d_model'])
    
    model_filename = latest_weights_file_path(config)
    if model_filename:
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
    else:
        print("No saved model found. Using untrained model.")
    
    return model, tokenizer_src, tokenizer_tgt

if __name__ == "__main__":
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, tokenizer_src, tokenizer_tgt = load_model(config)
    model.to(device)
    
    while True:
        phrase = input("\nEnter a phrase to translate (or 'q' to quit): ")
        if phrase.lower() == 'q':
            break
        
        translated = translate(phrase, model, tokenizer_src, tokenizer_tgt, device)
        print(f"\nTranslated: {translated}")