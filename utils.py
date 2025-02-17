import torch

from transformers import RobertaTokenizer

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return [tokenizer.cls_token_id] + ids

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(ids+add_ids)
    
    return torch.tensor(pad_ids)


def make_batch_roberta(sessions):
    batch_input, batch_labels, batch_speaker_tokens = [], [], []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        
        context_speaker, context, speaker_utt_history, emotion, sentiment = data
        speaker_utt_list = []
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
        for i in range(len(speaker_utt_history)-1):
            speaker_utt_list.append(encode_right_truncated(speaker_utt_history[i], roberta_tokenizer))
          
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))
        
        if len(label_list) > 3:
            label_ind = label_list.index(emotion)
        else:
            label_ind = label_list.index(sentiment)
        batch_labels.append(label_ind)        
        
        batch_speaker_tokens.append(padding(speaker_utt_list, roberta_tokenizer))
    
    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_labels, batch_speaker_tokens
