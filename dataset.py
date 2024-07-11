import torch
from torch.utils.data import Dataset
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.eos_token = tokenizer_tgt.eos_id()
        self.pad_token = tokenizer_tgt.pad_id()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text)
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text)

        enc_num_padding_tokens = max(0, self.seq_len - len(enc_input_tokens) - 1)
        dec_num_padding_tokens = max(0, self.seq_len - len(dec_input_tokens))

        # Add EOS to encoder input
        encoder_input = enc_input_tokens + [self.eos_token] + [self.pad_token] * enc_num_padding_tokens

        # Decoder input starts with a pad token
        decoder_input = [self.pad_token] + dec_input_tokens + [self.pad_token] * (dec_num_padding_tokens - 1)

        # Label includes EOS
        label = dec_input_tokens + [self.eos_token] + [self.pad_token] * (dec_num_padding_tokens - 1)

        # Convert to tensors
        encoder_input = torch.tensor(encoder_input, dtype=torch.long)
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0