import torch
import torch.nn as nn

def combine_tokens(org, prompt, rewrt, tokenizer):
    # inputs are list of tokens
    a = tokenizer.encode("Original Essay:", add_special_tokens=False)
    b = tokenizer.encode("Prompt:",add_special_tokens=False)
    c = tokenizer.encode("Rewritten Essay:",add_special_tokens=False)
    return a + org + b + prompt + c + rewrt

def create_batch_txt(original_texts, rewritten_texts, prompts, tokenizer):
    # all inputs should be list to token_ids
    # return the "flattened outter product" between text and prompt
    token_ids = []
    rewritten_text_len = []
    for org,rewrt in zip(original_texts,rewritten_texts):
        n = len(rewrt)
        for prompt in prompts:
            token_ids.append(torch.tensor(combine_tokens(org, prompt, rewrt, tokenizer),dtype=torch.long))
            tot_len = token_ids[-1].shape[0]
            rewritten_text_len.append((tot_len-n,tot_len)) # start and end of rewritten_texts
    return token_ids,rewritten_text_len

def gen_batch(token_ids, rewritten_text_len, batch_size, tokenizer):
    n = len(token_ids)
    for i in range(min(n//batch_size+1,n)):
        tokenList,rewt_lenList = token_ids[i*batch_size:i*batch_size+batch_size],rewritten_text_len[i*batch_size:i*batch_size+batch_size]
        inputs = torch.nn.utils.rnn.pad_sequence(tokenList, batch_first=True, padding_value=tokenizer.pad_token_id).to('cuda')
        yield inputs,rewt_lenList

def sort_and_recover_indices(token_ids):
    # sort inputs by len before batching to minimize padding
    # Get the index that sorts the sequence lengths
    seq_lens = [len(t) for t in token_ids]
    sort_index = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i])
    # Create an inverse index to recover the original order
    recover_index = [0] * len(seq_lens)
    for i, ix in enumerate(sort_index):
        recover_index[ix] = i
    return sort_index, recover_index

def get_list(list_, idx): return [list_[id_] for id_ in idx]

def gen_CE(token_ids, rewritten_text_len, batch_size, tokenizer, model):
    loss = nn.CrossEntropyLoss()
    CE_out = []
    sort_index, recover_index = sort_and_recover_indices(token_ids)
    token_ids, rewritten_text_len = get_list(token_ids,sort_index), get_list(rewritten_text_len,sort_index) # sort by lens
    with torch.no_grad():
        for inputs,rewt_len in gen_batch(token_ids, rewritten_text_len, batch_size, tokenizer):
            outs = model(inputs,output_attentions=False,output_hidden_states=False,use_cache=False).logits
            for i,o,(s,e) in zip(inputs,outs,rewt_len):
                # rewt_len is start and end of rewritten_texts
                # shift by one as k-th output logits has access to k-th input
                CE_out.append(loss(o[s:e-1], i[s+1:e]).item())
    return get_list(CE_out,recover_index) # unsort