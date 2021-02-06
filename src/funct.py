import re
import torch
import argparse
from model import DeepPunctuation, DeepPunctuationCRF
from config import *

pretrained_model = 'bert-base-multilingual-cased'

tokenizer = MODELS[pretrained_model][1].from_pretrained(pretrained_model)
token_style = MODELS[pretrained_model][3]

device = torch.device('cpu')

deep_punctuation = DeepPunctuation(pretrained_model, freeze_bert=False, lstm_dim=-1)
deep_punctuation.to(device)

def inference(model, text):
    deep_punctuation.load_state_dict(torch.load(model, map_location=torch.device('cpu')), strict=False)
    deep_punctuation.eval()

    text = re.sub(r"[,:\-–.!;?]", '', text)
    words_original_case = text.split()
    words = text.lower().split()

    word_pos = 0
    sequence_len = 256
    result = ""
    decode_idx = 0
    punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}

    while word_pos < len(words):
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y_mask = [0]

        while len(x) < sequence_len and word_pos < len(words):
            tokens = tokenizer.tokenize(words[word_pos])
            if len(tokens) + len(x) >= sequence_len:
                break
            else:
                for i in range(len(tokens) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                    y_mask.append(0)
                x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                y_mask.append(1)
                word_pos += 1
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y_mask.append(0)
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

        x = torch.tensor(x)
        y_mask = torch.tensor(y_mask)
        attn_mask = torch.tensor([attn_mask])
        x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

        with torch.no_grad():
            y_predict = deep_punctuation(x, attn_mask)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            y_predict = torch.argmax(y_predict, dim=1).view(-1)
        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                decode_idx += 1
    
    t = False
    for i in range(len(result)):
        if result[i] == '.':
            t = True
        if result[i] in 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя' and t:
            t = False
            result[i] = result[i].upper()
        
    return result
 
