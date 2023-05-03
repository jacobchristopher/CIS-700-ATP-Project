import torch as tr

def make_positional_encodings(max_len, d_model, denom_base):
    d2 = d_model // 2
    PE = tr.cat([
        trig(tr.arange(max_len).unsqueeze(1) / denom_base**(tr.arange(d2)/d2))
        for trig in (tr.sin, tr.cos)], dim=1)
    return PE

def trigonometric_positional_encoder(max_len, d_model, denom_base, device):
    PE = make_positional_encodings(max_len, d_model, denom_base)
    PE = PE.to(device)
    def encoder(inputs):
        return inputs.to("cuda") + PE[:inputs.shape[1]] # broadcast batch size, seq len, embedding dim
    return encoder