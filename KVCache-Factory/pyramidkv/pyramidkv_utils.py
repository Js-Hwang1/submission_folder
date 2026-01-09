import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import os

from typing import List


from typing import List, Optional, Tuple
from transformers.cache_utils import Cache

# =============================================================================
# Module-level debug logging for CircuitKV (singleton pattern)
# =============================================================================
_CIRCUITKV_DEBUG_LOG = None
_CIRCUITKV_DEBUG_INITIALIZED = False
_CIRCUITKV_SAMPLE_COUNTER = 0
_CIRCUITKV_LAYER_COUNTER = 0  # Track which layer we're on within a sample

def _get_circuitkv_debug_log():
    """Get or create the shared debug log file."""
    global _CIRCUITKV_DEBUG_LOG, _CIRCUITKV_DEBUG_INITIALIZED
    if not _CIRCUITKV_DEBUG_INITIALIZED:
        log_path = os.path.join(os.getcwd(), "longbench_CKV_dbg.log")
        _CIRCUITKV_DEBUG_LOG = open(log_path, "w")
        _CIRCUITKV_DEBUG_LOG.write("=" * 80 + "\n")
        _CIRCUITKV_DEBUG_LOG.write("CircuitKV Debug Log - Landmark Walker Analysis\n")
        _CIRCUITKV_DEBUG_LOG.write("=" * 80 + "\n\n")
        _CIRCUITKV_DEBUG_INITIALIZED = True
    return _CIRCUITKV_DEBUG_LOG

def _circuitkv_debug_next_sample():
    """Increment sample counter and reset layer counter."""
    global _CIRCUITKV_SAMPLE_COUNTER, _CIRCUITKV_LAYER_COUNTER
    _CIRCUITKV_SAMPLE_COUNTER += 1
    _CIRCUITKV_LAYER_COUNTER = 0
    return _CIRCUITKV_SAMPLE_COUNTER

def _circuitkv_debug_next_layer():
    """Increment layer counter."""
    global _CIRCUITKV_LAYER_COUNTER
    _CIRCUITKV_LAYER_COUNTER += 1
    return _CIRCUITKV_LAYER_COUNTER

def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, keep_idx, 1)                   
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1,-1,seqlen - recent_size,head_dim-k), kv_states[:, :, seqlen - recent_size:, :], ~mask

class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self) ->None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            import nvtx
            copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache


        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M 

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten()  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten()  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected] 
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] * all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(all_indices.shape[0], all_indices.shape[1], drop_len) # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat([selected_values, value_states[:, :, -window_size:, :]], dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge=="pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged, reduce='mean', include_self=True) # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')
        
    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent


class PyramidKVCluster():
    def __init__(self, num_hidden_layers = 32, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None, merge = None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
            
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
    
       
        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
        
        print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}")
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None, recent_size = 32, ratio =  0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.ratio = ratio
        self.recent_size = recent_size

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

    def update_think(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio)
            return kv_pruned, kv_recent, mask, value_states


class L2NormCluster():
    def __init__(self, max_capacity_prompt:int=256+64, layer_idx:int=0, skip_layers: List[int] = []):
        self.max_capacity_prompt = max_capacity_prompt
        self.layer_idx = layer_idx
        self.skip_layers = skip_layers

    def reset(self, max_capacity_prompt:int=256+64, layer_idx:int=0, skip_layers: List[int] = []):
        self.max_capacity_prompt = max_capacity_prompt
        self.layer_idx = layer_idx
        self.skip_layers = skip_layers
        
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"L2Norm max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif self.layer_idx in self.skip_layers:
            return key_states, value_states
        else:
            head_dim = key_states.size(-1)
            token_norms = torch.norm(key_states, p=2, dim=-1)
            sorted_indices = token_norms.squeeze(-1).argsort(dim=-1)
            sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            sorted_key_states = key_states.gather(dim=2, index=sorted_indices_expanded)
            sorted_value_states = value_states.gather(dim=2, index=sorted_indices_expanded)
            
            key_states = sorted_key_states[:, :, :self.max_capacity_prompt, :]
            value_states = sorted_value_states[:, :, :self.max_capacity_prompt, :]

            return key_states, value_states

class CAMKVCluster:
    def __init__(self, start_budget_ratio = 0.1, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.start_budget_ratio = start_budget_ratio
        self.merge = merge

    def reset(self, start_budget_ratio = 0.1, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.start_budget_ratio = start_budget_ratio
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"CAM max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum

            # merge recent tokens
            start_budget = math.ceil(self.start_budget_ratio * q_len)
            recent_budget = self.window_size
            # start_budget = math.ceil(self.start_budget_ratio * attn_weights.shape[-1])
            # recent_budget = math.ceil(self.recent_budget_ratio * attn_weights.shape[-1])
            # print(f"start_budget {start_budget}")
            # print(f"recent_budget {recent_budget}")

            # CAM merge
            seq_length = attn_weights.shape[-1]
            padding_length = 0
            merge_budget = recent_budget
            for token_index in range(start_budget + padding_length + recent_budget, seq_length):
                if token_index - recent_budget < 0 or token_index - recent_budget >= value_states.shape[2]:
                    continue
                attn_score = torch.mean(attn_weights[:, :, :token_index, :token_index], dim=-2)
                mean_attn = torch.max(torch.cat((attn_score[0, :, :start_budget], attn_score[0, :, token_index - recent_budget:token_index]), dim=-1), dim=-1)[0]
                merge_prob = attn_score[0, :, token_index - recent_budget] / mean_attn
                if torch.isnan(merge_prob).any(): merge_prob[torch.isnan(merge_prob)] = 0
                if torch.isinf(merge_prob).any(): merge_prob[torch.isinf(merge_prob)] = 1
                merge_mask = torch.bernoulli(merge_prob.clamp(min=0, max=1))
                score1 = value_states[:, :, token_index - recent_budget, ...].clone() * merge_mask.unsqueeze(-1) / merge_budget
                value_states[:, :, token_index - recent_budget + 1:token_index - recent_budget + merge_budget + 1, :] += score1.unsqueeze(2)

            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)

            return key_states, value_states


class H2OKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"H2O max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states


class StreamingLLMKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', merge = None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        print(f"StreamingLLM max_capacity_prompt {self.max_capacity_prompt}")
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class AdaKVCluster():
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',max_capacity_prompt=None,floor = None,normalize=None, layer_idx = None, num_hidden_layers=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = max_capacity_prompt - window_size
        self.floor_ratio = floor
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.num_hidden_layers = num_hidden_layers

        self.normalize = normalize
        self.layer_idx = layer_idx

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None


    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        sorted_attn_score,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        adaptive_attn_score = sorted_attn_score
        length = adaptive_attn_score.size(dim=-1)
        if self.normalize:
            ratio_weight = sorted_attn_score[...,:self.base_capacity].sum(dim=-1,keepdim=True)/sorted_attn_score.sum(dim=-1,keepdim=True)
            adaptive_attn_score = adaptive_attn_score*ratio_weight
        adaptive_attn_score = adaptive_attn_score.reshape(bsz,length*num_heads)
        sorted_indices = torch.topk(adaptive_attn_score,k=num_heads*self.base_capacity,dim=-1).indices
        sorted_indices = sorted_indices//length
        # floor capacity set
        head_adaptive_capacity = torch.zeros((bsz,num_heads),device=_device,dtype = sorted_indices.dtype)
        head_adaptive_capacity.scatter_add_(-1,sorted_indices,torch.ones_like(sorted_indices,dtype=head_adaptive_capacity.dtype),)
        assert head_adaptive_capacity.sum().item() == num_heads*self.base_capacity
        head_adaptive_capacity = torch.round(head_adaptive_capacity * (1-self.floor_ratio) + self.floor_capacity).int()
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads):
            cache_index = sorted_attn_score_indices[head_idx][...,:head_adaptive_capacity[0][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states,heads_value_states


class HeadKVCluster():
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''
    def __init__(self, window_size = 32, kernel_size = 7, pooling = 'maxpool',max_capacity_prompt=None, layer_idx = None, num_hidden_layers=None, head_capacity=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = max_capacity_prompt - window_size
        self.head_adaptive_capacity = head_capacity
        self.num_hidden_layers = num_hidden_layers

        self.layer_idx = layer_idx

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(
            head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(attn_weights_mean, kernel_size=self.kernel_size,
                                                     padding=self.kernel_size // 2,
                                                     stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self,  key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score= self.calcul_attn_sore(key_states,query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens, dim=0, dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat(
                [self.cu_klen, torch.tensor([self.klen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.layer_qlens = torch.ones(num_heads, dtype=torch.int32,device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat(
                [self.cu_qlen, torch.tensor([self.qlen_sum], dtype=torch.int32, device=_device)], dim=0)
            self.cu_offset = torch.arange(0, num_heads + 1, dtype=torch.int32, device=_device)
            self.cu_head_offset = torch.arange(1, num_heads+1, dtype=torch.int32, device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads, q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(-1, head_dim)

        # if you need to weight the attn_score
        _,sorted_attn_score_indices = attn_score.sort(dim=-1,descending=True)
        sorted_attn_score_indices = sorted_attn_score_indices.split(1,dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads):
            cache_index = sorted_attn_score_indices[head_idx][...,:self.head_adaptive_capacity[self.layer_idx][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1, 1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(dim=2,index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(dim=2,index=cache_index)
            selected_k = torch.cat([top_Kcache,origin_heads_key_states[head_idx][:, :, -self.window_size:, :]],dim=2)
            selected_v = torch.cat([top_Vcache,origin_heads_value_states[head_idx][:, :, -self.window_size:, :]],dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states,heads_value_states

def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = PyramidKVCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )
 
def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = SnapKVCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_think(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
        if not hasattr(self.config, 'recent_size'):
            self.config.recent_size = 32
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = 0.4
    
    
    self.kv_cluster = SnapKVCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        recent_size = self.config.recent_size,
        ratio = self.config.ratio
        )

def init_l2norm(self):
    
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'layer_idx'):
            self.config.layer_idx = 0
        if not hasattr(self.config, 'skip_layers'):
            self.config.skip_layers = [0,1]

    self.kv_cluster = L2NormCluster( 
        max_capacity_prompt = self.config.max_capacity_prompt,
        layer_idx = self.layer_idx,
        skip_layers = self.config.skip_layers
    )

def init_CAM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    
    
    self.kv_cluster = CAMKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_H2O(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    self.kv_cluster = H2OKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
    
    
    self.kv_cluster = StreamingLLMKVCluster(
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        merge = self.config.merge,
        )

def init_adakv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'floor_ratio'):
            self.config.floor_ratio = 0.2
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = True
    # max_capacity_prompt --> base_capacity
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = AdaKVCluster( 
            num_hidden_layers = self.config.num_hidden_layers,
            layer_idx = self.layer_idx,
            window_size = self.config.window_size, 
            max_capacity_prompt = self.config.max_capacity_prompt, 
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            floor = self.config.floor,
            normalize = self.config.normalize
            )


def init_headkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'head_capacity'):
            raise ValueError("Must have head_capacity")
    # max_capacity_prompt --> base_capacity
    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = HeadKVCluster(
            num_hidden_layers = self.config.num_hidden_layers,
            layer_idx = self.layer_idx,
            window_size = self.config.window_size,
            max_capacity_prompt = self.config.max_capacity_prompt,
            kernel_size = self.config.kernel_size,
            pooling = self.config.pooling,
            head_capacity=self.config.head_capacity
            )



# ============================================================================
# CircuitKV: Capacitive State-Space Model for KV Cache Eviction
# ============================================================================

class CircuitKVCluster():
    """
    CircuitKV v0.5.0: Landmark Absorbing Walker (Landmarks as Sources AND Sinks).

    This class implements multi-source absorbing walks where landmarks can BOTH
    source AND absorb walkers, creating a "mesh" of current flows between landmarks.

    Key Innovation (v0.5.0 - Landmark Absorbing):
    - STRATIFIED landmark selection: Divide sequence into segments, pick best H2O per segment
    - LANDMARKS AS SINKS: Walkers absorb at ANY landmark (not just tokens 0-3)
    - MESH NETWORK: Creates local current flows between adjacent landmarks
    - Captures "local bridges" - tokens that connect ADJACENT landmarks

    Physics Analogy:
    - Old (v0.4.0): Single battery (Query+ to Sink-)
    - New (v0.5.0): Mesh network where each landmark can source or sink current

    Why Landmark Absorbing > Standard Walker:
    - Standard: All walkers flow to tokens 0-3 (single sink)
    - Absorbing: Walkers stop at nearest landmark, capturing LOCAL importance
    - Bridge tokens between L1 and L2 get visits from BOTH directions

    Algorithm:
    1. Compute full attention matrix from query/key states
    2. STRATIFIED landmark selection: one landmark per segment, best H2O within segment
    3. Launch walkers from ALL sources (landmarks + query) in parallel
    4. Walkers ABSORB at any landmark (or sink tokens 0-3)
    5. Apply reachability normalization: score[p] = visits[p] / total_reachable[p]
    6. Select top tokens by normalized scores for KV cache retention

    Particularly effective for:
    - Multi-hop reasoning (HotpotQA, 2WikiMQA)
    - Long document QA (NarrativeQA, Qasper)
    - Cases where important tokens are LOCAL bridges between landmarks

    Configuration (v0.5.0 defaults):
    - num_landmarks: Number of diverse landmarks (default: 32)
    - min_spacing: Minimum segment size for stratified (default: 50)
    - walkers_per_source: Walkers launched from each source (default: 100)
    - query_boost: Weight multiplier for query-sourced walkers (default: 2.0)
    - absorb_at_landmarks: If True (default), landmarks absorb walkers (NEW behavior)
    """

    def __init__(
        self,
        window_size: int = 32,
        max_capacity_prompt: int = 2048,
        kernel_size: int = 5,
        pooling: str = 'avgpool',
        merge = None,
        # CircuitKV-specific parameters
        sink_size: int = 4,  # Absorbing boundary (first 4 tokens)
        top_k: int = 32,
        alpha: float = 0.85,  # Unused by CircuitKV, kept for API compatibility
        num_walkers: int = 1024,
        num_steps: int = 100,  # MAX_STEPS for safety timeout
        # Capacitive model parameters
        decay: float = 0.95,  # EMA decay for charge accumulation
        observation_window: int = 1,  # P0+P1: Last W tokens for H2O attention + multi-source walks
        # RC+B: Bidirectional Circuit Walks
        # NOTE: Disabled - R@65 improved but narrativeqa dropped (25.10 -> 23.78)
        bidirectional: bool = False,
        # Spectral + Walker + MAX mode (v0.2.0)
        use_combined_scoring: bool = False,  # False = Walker-only, True = Spectral + Walker + MAX
        num_power_iterations: int = 10,  # Power iterations for spectral
        # Landmark Walker parameters (v0.4.0 - Stratified + Reachability)
        num_landmarks: int = 32,  # Number of diverse landmarks (32 for comprehensive coverage)
        min_spacing: int = 50,  # Minimum segment size for stratified sampling
        walkers_per_source: int = 100,  # Walkers per source (landmark/query)
        query_boost: float = 2.0,  # Weight multiplier for query-sourced walkers
        position_alpha: float = 0.6,  # Exponent for positional normalization (unused in v0.5.0)
        use_reachability: bool = False,  # Unused in v0.5.0 (always uses reachability internally)
        # v0.5.0: Landmark Absorbing Walker
        absorb_at_landmarks: bool = True,  # True = landmarks absorb walkers (NEW), False = old behavior
        # Debug logging
        debug: bool = False,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.sink_size = sink_size
        self.top_k = top_k
        self.alpha = alpha
        self.num_walkers = num_walkers
        self.num_steps = num_steps
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.decay = decay
        self.observation_window = observation_window
        self.bidirectional = bidirectional
        self.use_combined_scoring = use_combined_scoring
        self.num_power_iterations = num_power_iterations
        # Landmark Walker
        self.num_landmarks = num_landmarks
        self.min_spacing = min_spacing
        self.walkers_per_source = walkers_per_source
        self.query_boost = query_boost
        self.position_alpha = position_alpha
        self.use_reachability = use_reachability
        # v0.5.0: Landmark Absorbing
        self.absorb_at_landmarks = absorb_at_landmarks
        # Debug
        self.debug = debug
        self._debug_log = None
        self._sample_counter = 0

        # Lazy initialization of CUDA graph
        self._graph = None
        self._device = None
        self._max_seq_len = 8192  # Will be updated on first use

        # Capacitive State: Accumulated charge for each token
        self._accumulated_charge = None
        self._h2o_scores = None  # Store H2O scores for MAX combination
        self._prefill_initialized = False

        # Initialize debug log file
        if self.debug:
            self._init_debug_log()

    def reset(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 2048,
        kernel_size: int = 5,
        pooling: str = 'avgpool',
        merge = None,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        if self._graph is not None:
            self._graph.reset()
        # Reset capacitive state
        self._accumulated_charge = None
        self._h2o_scores = None
        self._prefill_initialized = False

    def _init_debug_log(self):
        """Initialize debug log (uses module-level singleton)."""
        log = _get_circuitkv_debug_log()
        # Only write config on first init (layer 0)
        if _CIRCUITKV_LAYER_COUNTER == 0:
            log.write(f"Configuration:\n")
            log.write(f"  num_landmarks: {self.num_landmarks}\n")
            log.write(f"  min_spacing: {self.min_spacing}\n")
            log.write(f"  walkers_per_source: {self.walkers_per_source}\n")
            log.write(f"  query_boost: {self.query_boost}\n")
            log.write(f"  position_alpha: {self.position_alpha}\n")
            log.write(f"  max_capacity_prompt: {self.max_capacity_prompt}\n")
            log.write(f"  window_size: {self.window_size}\n")
            log.write(f"  sink_size: {self.sink_size}\n")
            log.write("\n" + "=" * 80 + "\n\n")
            log.flush()

    def _log_debug(
        self,
        q_len: int,
        h2o_scores: torch.Tensor,
        landmark_scores: torch.Tensor,
        keep_mask: torch.Tensor,
        full_attn: torch.Tensor,
    ):
        """Log detailed debug info for each eviction decision."""
        if not self.debug:
            return

        log = _get_circuitkv_debug_log()
        layer_num = _circuitkv_debug_next_layer()

        # Only log detailed info for layer 0 (to avoid overwhelming output)
        # Other layers just get a summary
        if layer_num == 1:
            # This is layer 0 (first call increments to 1)
            sample_num = _circuitkv_debug_next_sample()

            log.write(f"\n{'='*80}\n")
            log.write(f"SAMPLE #{sample_num} | seq_len={q_len}\n")
            log.write(f"{'='*80}\n\n")

            # 1. H2O Scores Analysis
            h2o_cpu = h2o_scores[:q_len].cpu()
            h2o_top_k = 20
            h2o_topk_vals, h2o_topk_idx = torch.topk(h2o_cpu, min(h2o_top_k, q_len))
            log.write(f"H2O SCORES (column sums - total incoming attention):\n")
            log.write(f"  Top {h2o_top_k} positions by H2O score:\n")
            for i, (pos, val) in enumerate(zip(h2o_topk_idx.tolist(), h2o_topk_vals.tolist())):
                log.write(f"    {i+1:2d}. pos={pos:5d}  h2o={val:.4f}\n")
            log.write(f"  H2O stats: min={h2o_cpu.min():.4f}, max={h2o_cpu.max():.4f}, mean={h2o_cpu.mean():.4f}\n\n")

            # 2. Landmark Walker Scores Analysis
            lw_cpu = landmark_scores[:q_len].cpu()
            lw_top_k = 30
            lw_topk_vals, lw_topk_idx = torch.topk(lw_cpu, min(lw_top_k, q_len))
            log.write(f"LANDMARK WALKER SCORES (reachability normalized):\n")
            log.write(f"  Top {lw_top_k} positions by LW score:\n")
            for i, (pos, val) in enumerate(zip(lw_topk_idx.tolist(), lw_topk_vals.tolist())):
                h2o_val = h2o_cpu[pos].item() if pos < len(h2o_cpu) else 0
                log.write(f"    {i+1:2d}. pos={pos:5d}  lw={val:.4f}  h2o={h2o_val:.4f}\n")
            log.write(f"  LW stats: min={lw_cpu.min():.4f}, max={lw_cpu.max():.4f}, mean={lw_cpu.mean():.4f}\n\n")

            # 3. Compare H2O vs LW rankings
            log.write(f"H2O vs LANDMARK WALKER COMPARISON:\n")
            h2o_rank = torch.argsort(h2o_cpu, descending=True)
            lw_rank = torch.argsort(lw_cpu, descending=True)

            # Find tokens that LW ranks much higher than H2O (bridge tokens!)
            h2o_rank_map = {pos.item(): rank for rank, pos in enumerate(h2o_rank)}
            lw_rank_map = {pos.item(): rank for rank, pos in enumerate(lw_rank)}

            rank_diffs = []
            for pos in range(q_len):
                h2o_r = h2o_rank_map.get(pos, q_len)
                lw_r = lw_rank_map.get(pos, q_len)
                rank_diffs.append((pos, h2o_r - lw_r, h2o_r, lw_r))  # positive = LW ranks higher

            rank_diffs.sort(key=lambda x: x[1], reverse=True)
            log.write(f"  Tokens LW ranks HIGHER than H2O (potential bridge tokens):\n")
            for pos, diff, h2o_r, lw_r in rank_diffs[:15]:
                if diff > 10:  # Only show significant differences
                    log.write(f"    pos={pos:5d}  h2o_rank={h2o_r:5d}  lw_rank={lw_r:5d}  diff={diff:+5d}\n")
            log.write("\n")

            # 4. Eviction Decision
            kept_positions = keep_mask.nonzero(as_tuple=True)[0].cpu().tolist()
            evicted_positions = (~keep_mask).nonzero(as_tuple=True)[0].cpu().tolist()

            log.write(f"EVICTION DECISION:\n")
            log.write(f"  Budget: {self.max_capacity_prompt} tokens\n")
            log.write(f"  Kept: {len(kept_positions)} tokens\n")
            log.write(f"  Evicted: {len(evicted_positions)} tokens\n")

            # Show top evicted tokens (ones we might regret evicting)
            evicted_with_scores = [(pos, lw_cpu[pos].item(), h2o_cpu[pos].item()) for pos in evicted_positions]
            evicted_with_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by LW score
            log.write(f"  Top 20 evicted tokens (highest LW score among evicted):\n")
            for pos, lw_val, h2o_val in evicted_with_scores[:20]:
                log.write(f"    pos={pos:5d}  lw={lw_val:.4f}  h2o={h2o_val:.4f}\n")
            log.write("\n")

            # 5. Position distribution of kept tokens
            kept_positions_sorted = sorted(kept_positions)
            n_early = sum(1 for p in kept_positions_sorted if p < q_len // 4)
            n_mid = sum(1 for p in kept_positions_sorted if q_len // 4 <= p < 3 * q_len // 4)
            n_late = sum(1 for p in kept_positions_sorted if p >= 3 * q_len // 4)
            log.write(f"POSITION DISTRIBUTION OF KEPT TOKENS:\n")
            log.write(f"  Early (0-25%): {n_early}\n")
            log.write(f"  Middle (25-75%): {n_mid}\n")
            log.write(f"  Late (75-100%): {n_late}\n")
            log.write("\n")

            log.flush()

    def _lazy_init(self, device, seq_len: int):
        """Initialize CUDA graph on first use."""
        if self._graph is None or self._device != device:
            try:
                from circuit_kv import _C as circuit_kv_cuda
                self._max_seq_len = max(seq_len * 2, 8192)  # Buffer for growth
                self._graph = circuit_kv_cuda.CircuitGraph(
                    self._max_seq_len,
                    self.top_k,
                    self.alpha,
                    self.num_walkers,
                    self.num_steps,
                )
                self._device = device
            except ImportError:
                raise ImportError(
                    "CircuitKV CUDA extension not found. "
                    "Install with: pip install -e ./CircuitKV"
                )

    def _get_keep_mask(self, scores: torch.Tensor, budget: int, seq_len: int) -> torch.Tensor:
        """
        Get a boolean mask indicating which tokens to keep.

        Implements Static Budgeting logic:
        1. Handle both static (int) and dynamic (float) budgets
        2. Always keep Sink tokens (first sink_size) and Local window (last window_size)
        3. Fill remaining budget with top current-flow scored tokens

        Args:
            scores: Current-flow scores from random walks [seq_len]
            budget: Total tokens to keep (static) or ratio (dynamic)
            seq_len: Current sequence length

        Returns:
            Boolean mask [seq_len] where True = keep
        """
        # 1. Handle Budget Types (Static vs Dynamic)
        if isinstance(budget, float) and budget <= 1.0:
            # Dynamic: 0.2 means 20% of sequence
            target_count = int(seq_len * budget)
        else:
            # Static: 1024 means exactly 1024 tokens
            target_count = int(budget)

        # 2. Safety Constraints: Always keep Sink + Local
        min_required = self.sink_size + self.window_size
        if target_count < min_required:
            target_count = min_required

        # 3. Circuit Selection
        device = scores.device
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[:self.sink_size] = True  # Keep Sink

        local_start = max(self.sink_size, seq_len - self.window_size)
        mask[local_start:] = True  # Keep Local

        current_kept = mask.sum().item()
        remaining_slots = max(0, target_count - current_kept)

        # Fill remaining budget with global top scores
        if remaining_slots > 0:
            # Zero out scores for tokens we already force-kept
            scores_masked = scores[:seq_len].clone()
            scores_masked[mask] = float('-inf')

            # Top-K selection for the remaining budget
            k = min(remaining_slots, (~mask).sum().item())
            if k > 0:
                _, topk_indices = torch.topk(scores_masked, k)
                mask[topk_indices] = True

        return mask

    def _initialize_charge_from_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        q_len: int,
        head_dim: int,
    ):
        """
        Initialize capacitor charge using PURE Walker scores (Absorbing Random Walk).

        No H2O mixing - just absorbing random walks from source toward sink.
        """
        import numpy as np

        device = key_states.device

        # Compute attention weights from window queries
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # Apply causal mask to window
        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -self.window_size:, -self.window_size:] += mask[None, None, :, :]

        # Softmax to get transition probabilities
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

        # Average across batch and heads: [window_size, seq_len]
        P = attn_weights.mean(dim=(0, 1))

        seq_len = P.shape[-1]
        non_window_len = max(0, seq_len - self.window_size)

        if non_window_len <= 0:
            self._accumulated_charge = torch.zeros(
                self._max_seq_len, device=device, dtype=torch.float32
            )
            self._prefill_initialized = True
            return

        # =====================================================================
        # PURE Absorbing Random Walk (no H2O mixing)
        # =====================================================================
        P_np = P.cpu().numpy()

        num_walkers = 1000
        sink_boundary = self.sink_size

        # Adaptive max_steps: ensure we can reach sink from any position
        # With adaptive jumps, we need ~20 steps to cross seq_len
        # Be conservative: allow 2x that for stochastic variance
        max_steps = max(100, seq_len // 20)

        # Visit counts - PURE walker signal
        walk_visits = np.zeros(seq_len, dtype=np.float32)

        # Last query's attention distribution (starting point for walks)
        last_q_attn = P_np[-1, :]

        if last_q_attn.sum() > 1e-8:
            start_probs = last_q_attn / last_q_attn.sum()

            for _ in range(num_walkers):
                # Start from a position sampled by last query's attention
                current = np.random.choice(seq_len, p=start_probs)

                for step in range(max_steps):
                    # Record visit (except sink tokens)
                    if current >= sink_boundary:
                        walk_visits[current] += 1.0

                    # Check if absorbed at sink
                    if current < sink_boundary:
                        break

                    # Transition based on position
                    if current >= seq_len - self.window_size:
                        # In window region: use actual attention
                        window_offset = current - (seq_len - self.window_size)
                        if 0 <= window_offset < self.window_size:
                            row_probs = P_np[window_offset, :]
                            if row_probs.sum() > 1e-8:
                                row_probs = row_probs / row_probs.sum()
                                current = np.random.choice(seq_len, p=row_probs)
                            else:
                                current = max(0, current - 1)
                        else:
                            current = max(0, current - 1)
                    else:
                        # Outside window: ADAPTIVE jump toward sink
                        # Jump size scales with distance to sink (faster when far, slower when close)
                        distance_to_sink = current - sink_boundary

                        if np.random.random() < 0.8:
                            # 80%: Adaptive jump - ~5% of remaining distance
                            jump = max(1, distance_to_sink // 20)
                            # Add some randomness: 0.5x to 1.5x the base jump
                            jump = max(1, int(jump * (0.5 + np.random.random())))
                            current = max(sink_boundary, current - jump)
                        else:
                            # 20%: Small step (allows fine-grained exploration near sink)
                            current = max(0, current - 1)

        # Convert to tensor and normalize
        walker_scores = torch.from_numpy(walk_visits).float()

        # Extract non-window portion
        walker_scores_nonwindow = walker_scores[:non_window_len]

        # Normalize to [0, 1]
        max_score = walker_scores_nonwindow.max()
        if max_score > 0:
            walker_scores_nonwindow = walker_scores_nonwindow / max_score

        # Initialize accumulated charge
        self._accumulated_charge = torch.zeros(
            self._max_seq_len, device=device, dtype=torch.float32
        )
        self._accumulated_charge[:non_window_len] = walker_scores_nonwindow.to(device)
        self._prefill_initialized = True

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask,
        num_key_value_groups: int,
    ):
        """
        Update KV cache using Stratified + Reachability eviction (v0.4.0).

        This implements multi-source absorbing walks from geographically-diverse landmarks:
        1. Compute full attention matrix from query/key states
        2. Run landmark walker (CUDA kernel):
           - STRATIFIED landmark selection: one per segment, best H2O within segment
           - Launch walkers from ALL sources (landmarks + query) in parallel
           - Apply REACHABILITY normalization: visits / total_walkers_that_could_reach
        3. EVICTION: Select top tokens by normalized landmark walker scores

        Args:
            key_states: [bsz, num_heads, seq_len, head_dim]
            query_states: [bsz, num_heads, seq_len, head_dim]
            value_states: [bsz, num_heads, seq_len, head_dim]
            attention_mask: Attention mask
            num_key_value_groups: Number of KV groups

        Returns:
            Compressed key_states, value_states
        """
        # Check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f"CircuitKV (LandmarkWalker) max_capacity_prompt {self.max_capacity_prompt}")

        # If sequence is shorter than budget, no eviction needed
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        # Initialize CUDA graph if needed
        self._lazy_init(key_states.device, q_len)

        # =====================================================================
        # STEP 1: Compute full attention matrix (averaged across heads)
        # =====================================================================
        # Use last window queries for attention computation
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # Apply causal mask to window portion
        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, :, -self.window_size:] += mask[None, None, :, :]

        # Softmax and average across batch and heads
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_avg = attn_weights.mean(dim=(0, 1))  # [window_size, seq_len]

        # Build full attention matrix approximation
        # For positions outside the window, use uniform causal attention
        full_attn = torch.zeros(q_len, q_len, device=key_states.device, dtype=torch.float32)

        # Fill the window portion with actual attention
        full_attn[-self.window_size:, :] = attn_avg

        # For positions outside window, use H2O-WEIGHTED transitions (not uniform)
        # This guides walkers toward high-attention tokens instead of random backward jumps
        n_prefix = q_len - self.window_size
        if n_prefix > 1:
            # Compute H2O scores (column sums) from window attention
            # This captures how much recent tokens attend to each position
            h2o_scores = attn_avg.sum(dim=0)  # [seq_len]
            h2o_prefix = h2o_scores[:n_prefix].clone()

            # Ensure positive scores (add small epsilon to avoid zeros)
            h2o_prefix = h2o_prefix.clamp(min=1e-6)

            # For row i, transition prob to j = h2o[j] / sum(h2o[0:i]) for j < i
            # This creates a valid probability distribution that sums to 1
            cumsum = h2o_prefix.cumsum(dim=0)  # [n_prefix]

            # denom[i] = sum(h2o[0:i]) = cumsum[i-1]
            denom = torch.zeros(n_prefix, device=key_states.device, dtype=torch.float32)
            denom[1:] = cumsum[:-1]
            denom[0] = 1.0  # Avoid div-by-zero (row 0 masked anyway)

            # Broadcast to create transition matrix
            # h2o_expanded[i,j] = h2o[j] for all i
            h2o_expanded = h2o_prefix.unsqueeze(0).expand(n_prefix, n_prefix)
            denom_expanded = denom.unsqueeze(1).expand(n_prefix, n_prefix)

            # P[i,j] = h2o[j] / sum(h2o[0:i])
            h2o_trans = h2o_expanded / (denom_expanded + 1e-8)

            # Apply causal mask (only attend to strictly previous positions)
            mask = torch.tril(torch.ones(n_prefix, n_prefix, device=key_states.device, dtype=torch.float32), diagonal=-1)
            full_attn[:n_prefix, :n_prefix] = h2o_trans * mask

        # =====================================================================
        # STEP 2: Run Landmark Absorbing Walker (CUDA kernel) - v0.5.0
        # =====================================================================
        current_idx = q_len - 1

        # v0.5.0: Use Landmark Absorbing Walker (landmarks as sources AND sinks)
        self._graph.update_and_step_landmark_absorbing_walker(
            full_attn.contiguous(),
            current_idx,
            self.num_landmarks,
            self.walkers_per_source,
            self.query_boost,
            self.min_spacing,
            self.absorb_at_landmarks,  # True = landmarks absorb walkers (NEW behavior)
        )

        # Get normalized scores from landmark absorbing walker (v0.5.0)
        scores = self._graph.get_landmark_absorbing_scores()

        # =====================================================================
        # STEP 3: EVICTION BASED ON LANDMARK WALKER SCORES
        # =====================================================================
        # Compute keep mask using static budgeting
        keep_mask = self._get_keep_mask(
            scores,
            self.max_capacity_prompt,
            q_len
        )

        # Debug logging
        if self.debug:
            # Compute H2O scores (column sums) for comparison
            h2o_scores = full_attn.sum(dim=0)
            self._log_debug(q_len, h2o_scores, scores, keep_mask, full_attn)

        # Apply eviction per head
        # Get indices of tokens to keep (excluding local window which is appended)
        non_local_mask = keep_mask.clone()
        non_local_mask[-self.window_size:] = False
        keep_indices = non_local_mask.nonzero(as_tuple=True)[0]

        # Number of non-local tokens to keep
        num_keep = keep_indices.shape[0]

        # Gather selected tokens
        indices_expanded = keep_indices.view(1, 1, -1, 1).expand(bsz, num_heads, -1, head_dim)

        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices_expanded[:, :, :num_keep, :])
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices_expanded[:, :, :num_keep, :])

        # Append local window
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]

        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        return key_states, value_states


def init_circuitkv(self):
    """Initialize CircuitKV cluster with Landmark Absorbing Walker (v0.5.0)."""
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 64
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
        # CircuitKV-specific defaults
        if not hasattr(self.config, 'sink_size'):
            self.config.sink_size = 4  # Absorbing boundary
        if not hasattr(self.config, 'top_k'):
            self.config.top_k = 32
        if not hasattr(self.config, 'alpha'):
            self.config.alpha = 0.85  # Unused, kept for API compatibility
        if not hasattr(self.config, 'num_walkers'):
            self.config.num_walkers = 2048
        if not hasattr(self.config, 'num_steps'):
            self.config.num_steps = 100  # MAX_STEPS for safety timeout
        # Capacitive model parameter
        if not hasattr(self.config, 'decay'):
            self.config.decay = 0.99  # EMA decay for charge accumulation
        # RC+B: Bidirectional Circuit Walks (disabled - hurt narrativeqa score)
        if not hasattr(self.config, 'bidirectional'):
            self.config.bidirectional = False
        # Landmark Walker parameters (v0.4.0 - Stratified + Reachability)
        if not hasattr(self.config, 'num_landmarks'):
            self.config.num_landmarks = 32  # 32 for comprehensive coverage
        if not hasattr(self.config, 'min_spacing'):
            self.config.min_spacing = 50  # Min segment size for stratified
        if not hasattr(self.config, 'walkers_per_source'):
            self.config.walkers_per_source = 100  # Walkers per source
        if not hasattr(self.config, 'query_boost'):
            self.config.query_boost = 2.0  # Weight for query-sourced walkers
        if not hasattr(self.config, 'position_alpha'):
            self.config.position_alpha = 0.6  # Exponent for positional normalization
        if not hasattr(self.config, 'use_reachability'):
            self.config.use_reachability = False  # Unused in v0.5.0
        # v0.5.0: Landmark Absorbing Walker
        if not hasattr(self.config, 'absorb_at_landmarks'):
            self.config.absorb_at_landmarks = True  # True = landmarks absorb walkers (NEW)
        # Debug logging
        if not hasattr(self.config, 'circuitkv_debug'):
            self.config.circuitkv_debug = False

    self.kv_cluster = CircuitKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        merge=self.config.merge,
        sink_size=self.config.sink_size,
        top_k=self.config.top_k,
        alpha=self.config.alpha,
        num_walkers=self.config.num_walkers,
        num_steps=self.config.num_steps,
        decay=self.config.decay,
        bidirectional=self.config.bidirectional,
        # Landmark Walker parameters
        num_landmarks=self.config.num_landmarks,
        min_spacing=self.config.min_spacing,
        walkers_per_source=self.config.walkers_per_source,
        query_boost=self.config.query_boost,
        position_alpha=self.config.position_alpha,
        use_reachability=self.config.use_reachability,
        # v0.5.0: Landmark Absorbing Walker
        absorb_at_landmarks=self.config.absorb_at_landmarks,
        # Debug
        debug=self.config.circuitkv_debug,
    )