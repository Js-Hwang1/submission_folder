"""Simple test to verify eviction is working"""
import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import types

MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
BUDGET, WINDOW, SINK = 2048, 64, 4

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
    attn_implementation='sdpa'
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded. Attention type: {type(model.model.layers[0].self_attn)}")

# Simple H2O atom function
def atom_h2o(attn, window, sink):
    return attn[:, :, :, :-window].sum(dim=-2).mean(dim=(0,1))

class DebugKVCluster:
    def __init__(self, budget, window, sink):
        self.budget = budget
        self.window = window
        self.sink = sink
        self.call_count = 0

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        self.call_count += 1
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f"\n[UPDATE_KV CALLED #{self.call_count}]")
        print(f"  Input shape: {key_states.shape}")
        print(f"  q_len={q_len}, budget={self.budget}")

        if q_len < self.budget:
            print(f"  -> Returning early (q_len < budget)")
            return key_states, value_states

        print(f"  -> EVICTING!")

        # Compute attention for scoring
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((self.window, self.window), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -self.window:, -self.window:] += mask[None, None, :, :]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # H2O scoring
        scores = atom_h2o(attn_weights, self.window, self.sink)

        # Select top tokens (exclude window)
        keep_n = self.budget - self.window - self.sink
        keep_mask = torch.zeros(q_len, dtype=torch.bool, device=key_states.device)
        keep_mask[:self.sink] = True

        middle_scores = scores[self.sink:]
        if len(middle_scores) > 0:
            top_k = min(keep_n, len(middle_scores))
            top_indices = middle_scores.topk(top_k).indices
            keep_mask[self.sink + top_indices] = True

        keep_mask[-self.window:] = True

        # Gather
        past_indices = keep_mask[:-self.window].nonzero(as_tuple=True)[0]
        past_indices_expanded = past_indices.view(1, 1, -1, 1).expand(bsz, num_heads, -1, head_dim)

        k_past = key_states[:, :, :-self.window, :].gather(dim=2, index=past_indices_expanded)
        v_past = value_states[:, :, :-self.window, :].gather(dim=2, index=past_indices_expanded)
        k_cur = key_states[:, :, -self.window:, :]
        v_cur = value_states[:, :, -self.window:, :]

        k_out = torch.cat([k_past, k_cur], dim=2)
        v_out = torch.cat([v_past, v_cur], dim=2)

        print(f"  Output shape: {k_out.shape}")

        return k_out, v_out

# Patch just layer 0 for debugging
layer = model.model.layers[0]
attn = layer.self_attn
attn.kv_cluster = DebugKVCluster(BUDGET, WINDOW, SINK)
attn.kv_seq_len = 0

original_forward = attn.forward
num_heads = attn.num_heads
head_dim = attn.head_dim
num_key_value_heads = attn.num_key_value_heads
num_key_value_groups = attn.num_key_value_groups
q_proj = attn.q_proj
k_proj = attn.k_proj
v_proj = attn.v_proj
o_proj = attn.o_proj
rotary_emb = model.model.rotary_emb

def patched_forward(self, hidden_states, attention_mask=None, position_ids=None,
                  past_key_value=None, output_attentions=False, use_cache=False,
                  cache_position=None, position_embeddings=None, **kwargs):
    bsz, q_len, _ = hidden_states.size()

    print(f"\n[FORWARD PASS]")
    print(f"  q_len={q_len}, use_cache={use_cache}")
    print(f"  past_key_value is None: {past_key_value is None}")
    if past_key_value is not None:
        print(f"  past_key_value type: {type(past_key_value)}")
        try:
            print(f"  cache length: {past_key_value.get_seq_length(0)}")
        except:
            print(f"  cache has no length")

    # QKV projection
    query_states = q_proj(hidden_states)
    key_states = k_proj(hidden_states)
    value_states = v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Compute kv_seq_len
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, 0)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, 0)

    print(f"  kv_seq_len={kv_seq_len}, self.kv_seq_len={self.kv_seq_len}")
    print(f"  key_states.shape[-2]={key_states.shape[-2]}")

    # Apply RoPE
    if position_embeddings is None:
        cos, sin = rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Expand KV
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    # EVICTION LOGIC
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        print(f"  Checking: key_states.shape[-2]={key_states.shape[-2]} == kv_seq_len={kv_seq_len}? {key_states.shape[-2] == kv_seq_len}")

        if key_states.shape[-2] == kv_seq_len:
            print(f"  -> PREFILL DETECTED! Calling update_kv...")
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, num_key_value_groups
            )
            past_key_value.update(key_states_compress, value_states_compress, 0, cache_kwargs)
            past_key_value._seen_tokens = self.kv_seq_len

            key_states = key_states_compress
            value_states = value_states_compress
        else:
            print(f"  -> GENERATION - using cache")
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, 0, cache_kwargs)
            past_key_value._seen_tokens = self.kv_seq_len

    # Compute attention
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = o_proj(attn_output)

    return attn_output, None, past_key_value

attn.forward = types.MethodType(patched_forward, attn)

print("\n" + "="*60)
print("TESTING GENERATION")
print("="*60)

# Reset before generation
attn.kv_seq_len = 0

prompt = "Hello, this is a test prompt. " * 200  # ~600 tokens
inp = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=3000).to('cuda')
print(f"\nInput tokens: {inp.input_ids.shape[1]}")

with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)

result = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nGenerated: {result}")
print(f"\nTotal update_kv calls: {attn.kv_cluster.call_count}")
