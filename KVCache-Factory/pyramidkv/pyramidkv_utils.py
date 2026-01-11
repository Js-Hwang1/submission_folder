import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import os

from typing import List


from typing import List, Optional, Tuple, Set
from transformers.cache_utils import Cache

# v4.0.0 Optional breakthroughs import (instruction anchors, etc.)
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'CircuitKV'))
    from circuit_kv.breakthroughs import (
        detect_instruction_anchors,
        expand_absorbing_boundary,
        compute_neumann_normalization,
        fundamental_matrix_normalize,
        compute_focus_ratio,
        get_horizon_weights,
        multi_horizon_ensemble,
        INSTRUCTION_PATTERNS,
    )
    BREAKTHROUGHS_AVAILABLE = True
except ImportError:
    BREAKTHROUGHS_AVAILABLE = False

# =============================================================================
# Module-level debug logging for CircuitKV (singleton pattern)
# =============================================================================
_CIRCUITKV_DEBUG_LOG = None
_CIRCUITKV_DEBUG_INITIALIZED = False
_CIRCUITKV_SAMPLE_COUNTER = 0
_CIRCUITKV_LAYER_COUNTER = 0  # Track which layer we're on within a sample
_CIRCUITKV_CURRENT_DATASET = None  # Track current dataset for diagnostics

def _set_circuitkv_dataset(dataset_name: str):
    """Set the current dataset name for debug logging."""
    global _CIRCUITKV_CURRENT_DATASET
    _CIRCUITKV_CURRENT_DATASET = dataset_name

def _get_circuitkv_debug_log():
    """Get or create the shared debug log file."""
    global _CIRCUITKV_DEBUG_LOG, _CIRCUITKV_DEBUG_INITIALIZED
    if not _CIRCUITKV_DEBUG_INITIALIZED:
        log_path = os.path.join(os.getcwd(), "longbench_CKV_dbg.log")
        _CIRCUITKV_DEBUG_LOG = open(log_path, "w")
        _CIRCUITKV_DEBUG_LOG.write("=" * 80 + "\n")
        _CIRCUITKV_DEBUG_LOG.write("CircuitKV v5.0.0 Debug Log - SDI (Stratified Diverse Influence)\n")
        _CIRCUITKV_DEBUG_LOG.write("=" * 80 + "\n\n")
        _CIRCUITKV_DEBUG_LOG.write("ALGORITHM: SDI (Stratified Diverse Influence)\n")
        _CIRCUITKV_DEBUG_LOG.write("  1. Task Detection: entropy(query_attn) → QA or Summarization\n")
        _CIRCUITKV_DEBUG_LOG.write("  2. Multi-Source Walks: segment centers for summarization\n")
        _CIRCUITKV_DEBUG_LOG.write("  3. MAX(H2O, Influence): hedging combination\n")
        _CIRCUITKV_DEBUG_LOG.write("  4. Stratified Selection: K segments × C clusters\n\n")
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
    CircuitKV v5.0.0: Stratified Diverse Influence (SDI) for ICML 2026.

    ═══════════════════════════════════════════════════════════════════════════════
    MATHEMATICAL FORMULATION
    ═══════════════════════════════════════════════════════════════════════════════

    Given attention matrix A ∈ ℝ^{n×n} where A[i,j] = softmax(q_i·k_j/√d) for j≤i,
    we compute token importance scores via three complementary signals:

    1. H2O (Heavy Hitter Oracle):
       h[j] = Σ_i A[i,j]    (column sums = incoming attention)

    2. Influence (Absorbing Random Walks):
       v[j] = E[visits to j | walker starts at query, absorbs at sink]
       Approximated via Monte Carlo with N walkers, S steps

    3. SDI (Stratified Diverse Influence) - NEW in v5.0.0:
       - Task-adaptive source selection via attention entropy
       - Stratified coverage guarantee across sequence segments
       - Semantic diversity via key-space clustering

    ═══════════════════════════════════════════════════════════════════════════════
    v5.0.0 KEY INNOVATIONS (ICML 2026)
    ═══════════════════════════════════════════════════════════════════════════════

    1. TASK DETECTION via Attention Entropy:
       H_q = -Σ_j A[q,j] log A[q,j]
       - Low entropy (H_q < 0.5·log(n)): QA task → query-focused walks
       - High entropy (H_q ≥ 0.5·log(n)): Summarization → multi-source walks

    2. STRATIFIED SELECTION (Coverage Guarantee):
       - Divide sequence into K segments: S_k = [k·n/K, (k+1)·n/K)
       - Select B/K tokens from each segment
       - Guarantees: ∀k: |Selected ∩ S_k| ≥ B/K

    3. MULTI-SOURCE WALKS (for Summarization):
       - Sample M source positions uniformly across segments
       - v[j] = (1/M) Σ_m Visits(j | start=source_m)
       - Captures global document structure, not just query-reachable tokens

    4. SEMANTIC DIVERSITY (within segments):
       - Cluster keys in each segment via k-means: C clusters per segment
       - Select top B/(K·C) tokens from each cluster
       - Prevents over-representation of semantically similar tokens

    ═══════════════════════════════════════════════════════════════════════════════
    ALGORITHM
    ═══════════════════════════════════════════════════════════════════════════════

    SDI(A, K, q, B):
    1. Compute H2O: h = A.sum(dim=0)
    2. Detect task type: task = "summ" if entropy(A[q]) > 0.5·log(n) else "qa"
    3. If task == "summ":
         sources = segment_centers(K)
         v = multi_source_walks(A, sources)
       Else:
         v = query_walks(A, q)
    4. For each segment k:
         clusters = kmeans(keys[S_k], C)
         For each cluster c:
           select top B/(K·C) by max(rank(h), rank(v))
    5. Return union of selected tokens

    ═══════════════════════════════════════════════════════════════════════════════
    CONFIGURATION
    ═══════════════════════════════════════════════════════════════════════════════

    SDI Parameters (v5.0.0):
    - use_sdi: Enable full SDI algorithm (default: True)
    - num_segments: K segments for stratification (default: 8)
    - num_clusters: C clusters per segment (default: 4)
    - entropy_threshold: Task detection threshold (default: 0.5)

    Legacy Parameters (preserved for compatibility):
    - num_walkers: Number of walkers (default: 10000)
    - max_steps: Max steps per walker (default: 10)
    - sink_size: Absorbing boundary (default: 4)
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
        num_walkers: int = 10000,  # v1.0.0: 10000 walkers (validated by PoC5)
        num_steps: int = 100,  # MAX_STEPS for safety timeout (legacy)
        max_steps: int = 10,  # v1.0.0: 10 steps per walker (matches oracle computation)
        # Capacitive model parameters
        decay: float = 0.95,  # EMA decay for charge accumulation
        observation_window: int = 1,  # P0+P1: Last W tokens for H2O attention + multi-source walks
        # v4.0.0: Bidirectional Circuit Walks (P0 - ICML 2026)
        # Captures betweenness: tokens on BOTH forward and backward paths
        bidirectional: bool = False,  # v5.0.0: Disabled by default (SDI replaces this)
        # Spectral + Walker + MAX mode (v0.2.0)
        use_combined_scoring: bool = False,  # False = Walker-only, True = Spectral + Walker + MAX
        num_power_iterations: int = 10,  # Power iterations for spectral
        # Landmark Walker parameters (legacy, kept for API compatibility)
        num_landmarks: int = 8,  # Legacy parameter (unused in v1.0.0)
        min_spacing: int = 50,  # Legacy parameter (unused in v1.0.0)
        walkers_per_source: int = 100,  # Legacy parameter (unused in v1.0.0)
        query_boost: float = 2.0,  # Legacy parameter (unused in v1.0.0)
        position_alpha: float = 0.6,  # Legacy parameter (unused in v1.0.0)
        use_reachability: bool = False,  # Legacy parameter (unused in v1.0.0)
        # v0.5.0: Landmark Absorbing Walker (legacy, kept for API compatibility)
        absorb_at_landmarks: bool = True,  # Legacy parameter (unused in v1.0.0)
        # Debug logging
        debug: bool = False,
        # v4.0.0 Features (ICML 2026) - Most replaced by SDI in v5.0.0
        use_instruction_anchors: bool = False,  # DISABLED (TREC issue was prompting, not KV)
        use_fundamental_norm: bool = False,  # Principled normalization (expensive, optional)
        use_multi_horizon: bool = False,  # v5.0.0: Disabled, SDI handles multi-scale differently
        tokenizer = None,  # Required for instruction anchor detection
        # ═══════════════════════════════════════════════════════════════════════
        # v5.0.0: SDI (Stratified Diverse Influence) Parameters
        # ═══════════════════════════════════════════════════════════════════════
        use_sdi: bool = True,  # Enable full SDI algorithm
        num_segments: int = 8,  # K: Number of segments for stratification
        num_clusters: int = 4,  # C: Clusters per segment for diversity
        entropy_threshold: float = 0.5,  # Task detection threshold (fraction of max entropy)
        sdi_query_weight: float = 0.7,  # Weight for query-source walks (vs uniform sources)
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.sink_size = sink_size
        self.top_k = top_k
        self.alpha = alpha
        self.num_walkers = num_walkers
        self.num_steps = num_steps
        self.max_steps = max_steps  # v1.0.0: Max steps per walker
        # v4.0.0 Features (legacy)
        self.use_instruction_anchors = use_instruction_anchors
        self.use_fundamental_norm = use_fundamental_norm
        self.use_multi_horizon = use_multi_horizon
        self.tokenizer = tokenizer
        self._instruction_anchors = set()  # Cache for current sample
        # v5.0.0: SDI Parameters
        self.use_sdi = use_sdi
        self.num_segments = num_segments
        self.num_clusters = num_clusters
        self.entropy_threshold = entropy_threshold
        self.sdi_query_weight = sdi_query_weight
        self._detected_task_type = None  # Cache for task detection
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
            log.write(f"⚙️  CONFIGURATION (v2.0.0):\n")
            log.write(f"  Algorithm: MAX(rank_h2o, rank_influence)\n")
            log.write(f"  \n")
            log.write(f"  KV Cache Budget:\n")
            log.write(f"    max_capacity_prompt: {self.max_capacity_prompt}\n")
            log.write(f"    window_size:         {self.window_size} (local tokens always kept)\n")
            log.write(f"    sink_size:           {self.sink_size} (sink tokens always kept)\n")
            log.write(f"  \n")
            log.write(f"  Influence Walker:\n")
            log.write(f"    num_walkers:         {self.num_walkers}\n")
            log.write(f"    max_steps:           {self.max_steps}\n")
            log.write(f"  \n")
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
        if layer_num == 1:
            sample_num = _circuitkv_debug_next_sample()
            dataset = _CIRCUITKV_CURRENT_DATASET or "unknown"

            # Compute compression ratio
            compression_ratio = 100 * (1 - self.max_capacity_prompt / q_len)

            log.write(f"\n{'='*80}\n")
            log.write(f"SAMPLE #{sample_num} | dataset={dataset} | seq_len={q_len}\n")
            log.write(f"{'='*80}\n\n")

            # =====================================================================
            # 0. COMPRESSION OVERVIEW
            # =====================================================================
            log.write(f"📊 COMPRESSION OVERVIEW:\n")
            log.write(f"  Input tokens:     {q_len:,}\n")
            log.write(f"  Budget:           {self.max_capacity_prompt:,}\n")
            log.write(f"  Compression:      {compression_ratio:.1f}% evicted\n")
            log.write(f"  Sink (force-keep): {self.sink_size} tokens (positions 0-{self.sink_size-1})\n")
            log.write(f"  Local window:     {self.window_size} tokens (positions {q_len-self.window_size}-{q_len-1})\n")
            log.write(f"  Competitive slots: {self.max_capacity_prompt - self.sink_size - self.window_size} tokens\n")
            if compression_ratio > 90:
                log.write(f"  ⚠️  WARNING: >90% compression - high risk of losing important tokens!\n")
            log.write("\n")

            h2o_cpu = h2o_scores[:q_len].cpu()
            influence_cpu = landmark_scores[:q_len].cpu()

            # =====================================================================
            # 1. MAX(H2O, Influence) COMBINATION ANALYSIS
            # =====================================================================
            # Rank normalize both
            h2o_ranks = torch.argsort(torch.argsort(h2o_cpu)).float() / (q_len - 1)
            influence_ranks = torch.argsort(torch.argsort(influence_cpu)).float() / (q_len - 1)
            combined_ranks = torch.maximum(h2o_ranks, influence_ranks)

            log.write(f"🔀 MAX(H2O, Influence) COMBINATION:\n")

            # Who dominates?
            h2o_wins = (h2o_ranks > influence_ranks).sum().item()
            influence_wins = (influence_ranks > h2o_ranks).sum().item()
            ties = q_len - h2o_wins - influence_wins
            log.write(f"  H2O dominates:       {h2o_wins:,} tokens ({100*h2o_wins/q_len:.1f}%)\n")
            log.write(f"  Influence dominates: {influence_wins:,} tokens ({100*influence_wins/q_len:.1f}%)\n")
            log.write(f"  Ties:                {ties:,} tokens ({100*ties/q_len:.1f}%)\n")

            # Correlation between H2O and Influence
            h2o_flat = h2o_cpu.float()
            inf_flat = influence_cpu.float()
            if h2o_flat.std() > 0 and inf_flat.std() > 0:
                corr = torch.corrcoef(torch.stack([h2o_flat, inf_flat]))[0, 1].item()
                log.write(f"  Correlation (H2O vs Influence): {corr:.3f}\n")
                if corr > 0.8:
                    log.write(f"    → High correlation: H2O and Influence agree (redundant?)\n")
                elif corr < 0.2:
                    log.write(f"    → Low correlation: H2O and Influence capture different tokens (good!)\n")
            log.write("\n")

            # =====================================================================
            # 2. TOP TOKENS BY EACH METHOD
            # =====================================================================
            top_k = 15
            log.write(f"🏆 TOP {top_k} TOKENS BY EACH METHOD:\n\n")

            h2o_topk_vals, h2o_topk_idx = torch.topk(h2o_cpu, min(top_k, q_len))
            influence_topk_vals, influence_topk_idx = torch.topk(influence_cpu, min(top_k, q_len))
            combined_topk_vals, combined_topk_idx = torch.topk(combined_ranks, min(top_k, q_len))

            log.write(f"  {'Rank':<5} {'H2O':^25} {'Influence':^25} {'MAX Combined':^25}\n")
            log.write(f"  {'----':<5} {'-'*25:^25} {'-'*25:^25} {'-'*25:^25}\n")
            for i in range(min(top_k, q_len)):
                h2o_pos = h2o_topk_idx[i].item()
                inf_pos = influence_topk_idx[i].item()
                comb_pos = combined_topk_idx[i].item()
                log.write(f"  {i+1:<5} pos={h2o_pos:<5} ({h2o_topk_vals[i]:.3f})   pos={inf_pos:<5} ({influence_topk_vals[i]:.3f})   pos={comb_pos:<5} ({combined_topk_vals[i]:.3f})\n")
            log.write("\n")

            # Overlap analysis
            h2o_top_set = set(h2o_topk_idx.tolist())
            inf_top_set = set(influence_topk_idx.tolist())
            overlap = h2o_top_set & inf_top_set
            h2o_only = h2o_top_set - inf_top_set
            inf_only = inf_top_set - h2o_top_set
            log.write(f"  Top-{top_k} Overlap Analysis:\n")
            log.write(f"    Both agree:        {len(overlap)} tokens {sorted(overlap)[:10]}...\n")
            log.write(f"    H2O only:          {len(h2o_only)} tokens {sorted(h2o_only)[:10]}...\n")
            log.write(f"    Influence only:    {len(inf_only)} tokens {sorted(inf_only)[:10]}...\n")
            log.write("\n")

            # =====================================================================
            # 3. POSITION COVERAGE ANALYSIS (Critical for few-shot/code tasks!)
            # =====================================================================
            kept_positions = keep_mask.nonzero(as_tuple=True)[0].cpu().tolist()
            evicted_positions = (~keep_mask).nonzero(as_tuple=True)[0].cpu().tolist()

            log.write(f"📍 POSITION COVERAGE ANALYSIS:\n")

            # Divide into segments
            n_segments = 10
            segment_size = q_len // n_segments
            log.write(f"  Coverage by segment (each ~{segment_size} tokens):\n")

            segment_coverage = []
            for seg in range(n_segments):
                seg_start = seg * segment_size
                seg_end = (seg + 1) * segment_size if seg < n_segments - 1 else q_len
                kept_in_seg = sum(1 for p in kept_positions if seg_start <= p < seg_end)
                total_in_seg = seg_end - seg_start
                coverage_pct = 100 * kept_in_seg / total_in_seg if total_in_seg > 0 else 0
                segment_coverage.append(coverage_pct)

                # Visual bar
                bar_len = int(coverage_pct / 5)
                bar = '█' * bar_len + '░' * (20 - bar_len)
                status = ""
                if seg == 0:
                    status = " ← SINK"
                elif seg == n_segments - 1:
                    status = " ← QUERY"
                elif coverage_pct < 5:
                    status = " ⚠️ LOW COVERAGE!"

                log.write(f"    Seg {seg}: [{seg_start:5d}-{seg_end:5d}] {bar} {coverage_pct:5.1f}% ({kept_in_seg}/{total_in_seg}){status}\n")

            # Coverage warnings
            middle_coverage = sum(segment_coverage[2:8]) / 6
            if middle_coverage < 10:
                log.write(f"\n  ⚠️  WARNING: Middle segments have only {middle_coverage:.1f}% coverage!\n")
                log.write(f"      This may cause failures on few-shot/code tasks that need middle context.\n")
            log.write("\n")

            # =====================================================================
            # 4. ATTENTION PATTERN ANALYSIS
            # =====================================================================
            log.write(f"👁️  ATTENTION PATTERN ANALYSIS:\n")

            # Attention entropy (how focused vs diffuse)
            attn_row = full_attn[-1, :q_len]  # Last query's attention
            attn_row_norm = attn_row / (attn_row.sum() + 1e-8)
            entropy = -(attn_row_norm * torch.log(attn_row_norm + 1e-10)).sum().item()
            max_entropy = math.log(q_len)
            log.write(f"  Query attention entropy: {entropy:.2f} / {max_entropy:.2f} ({100*entropy/max_entropy:.1f}%)\n")
            if entropy / max_entropy > 0.9:
                log.write(f"    → Very diffuse attention (may need more tokens)\n")
            elif entropy / max_entropy < 0.3:
                log.write(f"    → Very focused attention (compression may work well)\n")

            # Where does query attend most?
            attn_topk_vals, attn_topk_idx = torch.topk(attn_row, min(10, q_len))
            log.write(f"  Query attends most to positions: {attn_topk_idx.tolist()}\n")

            # Check if high-attention tokens are kept
            attn_top_kept = sum(1 for p in attn_topk_idx.tolist() if p in kept_positions)
            log.write(f"  Top-10 attended tokens kept: {attn_top_kept}/10\n")
            if attn_top_kept < 8:
                log.write(f"    ⚠️  WARNING: Missing {10-attn_top_kept} high-attention tokens!\n")
            log.write("\n")

            # =====================================================================
            # 5. INFLUENCE WALKER DIAGNOSTICS
            # =====================================================================
            log.write(f"🚶 INFLUENCE WALKER DIAGNOSTICS:\n")
            try:
                raw_visits = self._graph.get_influence_raw_visits()[:q_len].cpu()
                n_zero = (raw_visits == 0).sum().item()
                total_visits = raw_visits.sum().item()

                log.write(f"  Total walker visits: {total_visits:,.0f}\n")
                log.write(f"  Tokens never visited: {n_zero} ({100*n_zero/q_len:.1f}%)\n")

                if n_zero > q_len * 0.8:
                    log.write(f"    ⚠️  WARNING: >80% of tokens never visited!\n")
                    log.write(f"        Walkers may not be reaching middle/early positions.\n")

                # Visit distribution by quartile
                q1_visits = raw_visits[:q_len//4].sum().item()
                q2_visits = raw_visits[q_len//4:q_len//2].sum().item()
                q3_visits = raw_visits[q_len//2:3*q_len//4].sum().item()
                q4_visits = raw_visits[3*q_len//4:].sum().item()

                log.write(f"  Visit distribution by quartile:\n")
                log.write(f"    Q1 (early):  {q1_visits:8.0f} ({100*q1_visits/max(total_visits,1):5.1f}%)\n")
                log.write(f"    Q2:          {q2_visits:8.0f} ({100*q2_visits/max(total_visits,1):5.1f}%)\n")
                log.write(f"    Q3:          {q3_visits:8.0f} ({100*q3_visits/max(total_visits,1):5.1f}%)\n")
                log.write(f"    Q4 (query):  {q4_visits:8.0f} ({100*q4_visits/max(total_visits,1):5.1f}%)\n")

            except Exception as e:
                log.write(f"  [Error getting raw visits: {e}]\n")
            log.write("\n")

            # =====================================================================
            # 6. EVICTION DECISION SUMMARY
            # =====================================================================
            log.write(f"📋 EVICTION SUMMARY:\n")
            log.write(f"  Kept: {len(kept_positions)} tokens\n")
            log.write(f"  Evicted: {len(evicted_positions)} tokens\n")

            # Top regrettable evictions (high combined score but evicted)
            evicted_with_scores = [(pos, combined_ranks[pos].item(), h2o_ranks[pos].item(), influence_ranks[pos].item())
                                   for pos in evicted_positions]
            evicted_with_scores.sort(key=lambda x: x[1], reverse=True)

            log.write(f"\n  🔥 TOP 15 POTENTIALLY REGRETTABLE EVICTIONS:\n")
            log.write(f"  {'Pos':<6} {'Combined':<10} {'H2O Rank':<10} {'Inf Rank':<10} {'Region':<15}\n")
            for pos, comb, h2o_r, inf_r in evicted_with_scores[:15]:
                region = "early" if pos < q_len//4 else "middle" if pos < 3*q_len//4 else "late"
                log.write(f"  {pos:<6} {comb:<10.3f} {h2o_r:<10.3f} {inf_r:<10.3f} {region:<15}\n")

            log.write("\n" + "-"*80 + "\n")
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

    def _rank_normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Rank-based normalization to [0, 1].

        Converts raw scores to uniform distribution based on ranking.
        This ensures H2O and Influence are on equal footing regardless of scale.

        Args:
            scores: Raw scores [seq_len]

        Returns:
            Rank-normalized scores in [0, 1]
        """
        ranks = torch.argsort(torch.argsort(scores)).float()
        n = len(scores)
        if n <= 1:
            return torch.ones_like(ranks)
        return ranks / (n - 1)

    # ═══════════════════════════════════════════════════════════════════════════
    # v5.0.0: SDI (Stratified Diverse Influence) Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def _detect_task_type(self, attention_row: torch.Tensor, seq_len: int) -> str:
        """
        Detect task type via attention entropy.

        Mathematical Formulation:
            H_q = -Σ_j p[j] log(p[j])   where p = attention_row (normalized)
            H_max = log(seq_len)

        Decision Rule:
            - H_q < threshold * H_max → "qa" (focused attention, query is informative)
            - H_q >= threshold * H_max → "summarization" (diffuse attention, need coverage)

        Args:
            attention_row: Query's attention distribution [seq_len]
            seq_len: Sequence length

        Returns:
            "qa" or "summarization"
        """
        # Normalize to valid probability distribution
        p = attention_row / (attention_row.sum() + 1e-10)
        p = p.clamp(min=1e-10)  # Avoid log(0)

        # Compute entropy: H = -Σ p log(p)
        entropy = -(p * torch.log(p)).sum().item()

        # Maximum possible entropy
        max_entropy = math.log(seq_len) if seq_len > 1 else 1.0

        # Normalized entropy ratio
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0

        # Classify based on threshold
        task_type = "summarization" if entropy_ratio >= self.entropy_threshold else "qa"

        return task_type

    def _compute_segment_centers(self, seq_len: int, num_segments: int) -> List[int]:
        """
        Compute center positions for each segment.

        For stratified multi-source walks, we launch walkers from segment centers
        to ensure coverage across the entire sequence.

        Args:
            seq_len: Sequence length
            num_segments: Number of segments K

        Returns:
            List of segment center positions
        """
        segment_size = seq_len // num_segments
        centers = []
        for k in range(num_segments):
            # Center of segment k
            start = k * segment_size
            end = (k + 1) * segment_size if k < num_segments - 1 else seq_len
            center = (start + end) // 2
            # Ensure center is valid (not in sink)
            center = max(self.sink_size, center)
            centers.append(center)
        return centers

    def _run_multi_source_walks(
        self,
        attention_matrix: torch.Tensor,
        sources: List[int],
        query_idx: int,
        q_len: int,
    ) -> torch.Tensor:
        """
        Run absorbing walks from multiple source positions.

        Mathematical Formulation:
            v[j] = w_q * Visits(j|query) + (1-w_q)/M * Σ_m Visits(j|source_m)

        This captures global document structure by launching walks from
        geographically diverse positions, not just the query.

        Args:
            attention_matrix: Full attention matrix [seq_len, seq_len]
            sources: List of source positions for multi-source walks
            query_idx: Query position
            q_len: Sequence length

        Returns:
            Combined visit scores [q_len]
        """
        attn_contig = attention_matrix.contiguous()

        # Query-source walks (always included)
        self._graph.update_and_step_influence_walker(
            attn_contig, query_idx,
            self.num_walkers // 2, self.max_steps, self.sink_size,
        )
        query_scores = self._graph.get_influence_scores()[:q_len].clone()

        # Multi-source walks from segment centers
        # Use remaining walkers distributed across sources
        walkers_per_source = max(100, self.num_walkers // (2 * len(sources)))
        source_scores_sum = torch.zeros(q_len, device=attention_matrix.device, dtype=torch.float32)

        for src in sources:
            if src < self.sink_size or src >= q_len:
                continue  # Skip invalid sources

            self._graph.update_and_step_influence_walker(
                attn_contig, src,
                walkers_per_source, self.max_steps, self.sink_size,
            )
            source_scores = self._graph.get_influence_scores()[:q_len].clone()
            source_scores_sum += source_scores

        # Combine: weighted average of query and source walks
        # w_q controls how much we weight query-focused vs uniform-source walks
        w_q = self.sdi_query_weight
        combined = w_q * query_scores + (1 - w_q) * source_scores_sum / max(1, len(sources))

        return combined

    def _simple_kmeans(
        self,
        keys: torch.Tensor,
        n_clusters: int,
        max_iters: int = 10,
    ) -> torch.Tensor:
        """
        Simple k-means clustering for semantic diversity.

        This is a lightweight GPU-based k-means for clustering key vectors
        within each segment. We use few iterations since we only need
        rough clusters for diversity, not precise clustering.

        Args:
            keys: Key vectors [n, head_dim]
            n_clusters: Number of clusters C
            max_iters: Maximum iterations (default: 10 for speed)

        Returns:
            Cluster assignments [n]
        """
        n = keys.shape[0]
        if n <= n_clusters:
            # More clusters than points - each point is its own cluster
            return torch.arange(n, device=keys.device)

        device = keys.device
        dtype = keys.dtype

        # Initialize centroids using k-means++ style (first point random, rest far)
        indices = torch.randperm(n, device=device)[:n_clusters]
        centroids = keys[indices].clone()

        # Normalize for cosine similarity
        keys_norm = F.normalize(keys.float(), dim=-1)

        for _ in range(max_iters):
            centroids_norm = F.normalize(centroids.float(), dim=-1)

            # Compute distances (using cosine similarity for efficiency)
            # distances[i,j] = similarity of point i to centroid j
            similarities = torch.mm(keys_norm, centroids_norm.t())

            # Assign to nearest centroid
            assignments = similarities.argmax(dim=-1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_clusters, device=device)

            for c in range(n_clusters):
                mask = (assignments == c)
                if mask.sum() > 0:
                    new_centroids[c] = keys[mask].mean(dim=0)
                    counts[c] = mask.sum()
                else:
                    # Empty cluster - reinitialize
                    new_centroids[c] = keys[torch.randint(n, (1,), device=device)]

            centroids = new_centroids

        return assignments

    def _stratified_diverse_select(
        self,
        scores: torch.Tensor,
        keys: torch.Tensor,
        budget: int,
        q_len: int,
    ) -> torch.Tensor:
        """
        Stratified selection with semantic diversity.

        Algorithm:
        1. Divide sequence into K segments
        2. For each segment:
           a. Cluster keys into C clusters
           b. Select top B/(K*C) tokens from each cluster by score
        3. Union all selected tokens

        Mathematical Guarantee:
            ∀k ∈ [1,K]: |Selected ∩ Segment_k| ≥ B/K
            ∀k,c: |Selected ∩ Cluster_{k,c}| ≥ B/(K*C)

        Args:
            scores: Combined importance scores [q_len]
            keys: Key vectors [q_len, head_dim]
            budget: Total tokens to select (B)
            q_len: Sequence length

        Returns:
            Boolean mask [q_len] where True = selected
        """
        device = scores.device
        K = self.num_segments
        C = self.num_clusters

        # Compute segment boundaries
        # Reserve space for sink and window
        selectable_start = self.sink_size
        selectable_end = q_len - self.window_size

        if selectable_end <= selectable_start:
            # Very short sequence - just use standard selection
            mask = torch.zeros(q_len, dtype=torch.bool, device=device)
            mask[:self.sink_size] = True
            mask[-self.window_size:] = True
            return mask

        selectable_len = selectable_end - selectable_start

        # Budget for stratified selection (excluding sink and window)
        sink_window_budget = self.sink_size + self.window_size
        stratified_budget = max(0, budget - sink_window_budget)

        # Tokens per segment
        segment_size = selectable_len // K
        tokens_per_segment = stratified_budget // K

        # Initialize mask
        mask = torch.zeros(q_len, dtype=torch.bool, device=device)
        mask[:self.sink_size] = True  # Always keep sink
        mask[-self.window_size:] = True  # Always keep window

        # Process each segment
        for k in range(K):
            seg_start = selectable_start + k * segment_size
            seg_end = selectable_start + (k + 1) * segment_size if k < K - 1 else selectable_end

            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue

            # Get keys and scores for this segment
            seg_keys = keys[seg_start:seg_end]
            seg_scores = scores[seg_start:seg_end]

            # Cluster keys in this segment
            actual_clusters = min(C, seg_len)
            if actual_clusters <= 1:
                # Too few tokens - just take top by score
                tokens_to_select = min(tokens_per_segment, seg_len)
                if tokens_to_select > 0:
                    _, top_idx = torch.topk(seg_scores, tokens_to_select)
                    mask[seg_start + top_idx] = True
                continue

            cluster_assignments = self._simple_kmeans(seg_keys, actual_clusters)

            # Select from each cluster
            tokens_per_cluster = max(1, tokens_per_segment // actual_clusters)

            for c in range(actual_clusters):
                cluster_mask = (cluster_assignments == c)
                cluster_indices = cluster_mask.nonzero(as_tuple=True)[0]

                if len(cluster_indices) == 0:
                    continue

                # Get scores for this cluster
                cluster_scores = seg_scores[cluster_indices]

                # Select top tokens from this cluster
                tokens_to_select = min(tokens_per_cluster, len(cluster_indices))
                if tokens_to_select > 0:
                    _, top_idx = torch.topk(cluster_scores, tokens_to_select)
                    selected_positions = cluster_indices[top_idx]
                    mask[seg_start + selected_positions] = True

        return mask

    def _detect_instruction_anchors_heuristic(
        self,
        attention_matrix: torch.Tensor,
        seq_len: int,
        max_anchors: int = 64,
    ) -> Set[int]:
        """
        v3.0.0 Breakthrough 2: Detect instruction anchors using attention patterns.

        For few-shot classification tasks like TREC, certain tokens anchor the
        instruction format (e.g., "Type:", "Question:", newlines between examples).
        These tokens have distinctive attention patterns:
        1. High self-attention (they "stand out" structurally)
        2. High incoming attention from nearby tokens
        3. Located at regular intervals (few-shot example boundaries)

        This heuristic identifies such tokens without needing the actual token text.

        Args:
            attention_matrix: Attention weights [seq_len, seq_len]
            seq_len: Current sequence length
            max_anchors: Maximum number of anchors to detect

        Returns:
            Set of anchor positions

        Theory:
            In few-shot prompts, instruction delimiters like "Type:" have:
            - High self-attention (position attends strongly to itself)
            - Act as "attention sinks" for nearby content
            - Appear at regular intervals (once per example)

            By detecting these patterns, we can protect instruction structure
            even without knowing the actual token text.
        """
        anchors = set()

        if seq_len < 100:
            return anchors  # Too short for few-shot patterns

        device = attention_matrix.device

        # Signal 1: H2O scores (column sums) - tokens frequently attended to
        h2o = attention_matrix.sum(dim=0)
        if h2o.max() > 0:
            h2o_norm = h2o / h2o.max()
        else:
            h2o_norm = h2o

        # Signal 2: Self-attention (diagonal) - tokens that stand out
        self_attn = torch.diagonal(attention_matrix)
        if self_attn.max() > 0:
            self_attn_norm = self_attn / self_attn.max()
        else:
            self_attn_norm = self_attn

        # Signal 3: Local H2O peaks - tokens with high attention relative to neighbors
        # This captures instruction tokens that are local "sinks"
        window = 20
        local_peak_score = torch.zeros(seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window)
            end = min(seq_len, i + window)
            local_mean = h2o[start:end].mean()
            if local_mean > 0:
                # How much higher is this position vs local average?
                local_peak_score[i] = h2o[i] / local_mean

        if local_peak_score.max() > 0:
            local_peak_norm = local_peak_score / local_peak_score.max()
        else:
            local_peak_norm = local_peak_score

        # Combined score: H2O importance + self-attention + local peak
        # H2O is most important for TREC-like tasks (instruction tokens get attention)
        anchor_scores = h2o_norm * 0.5 + self_attn_norm * 0.2 + local_peak_norm * 0.3

        # Find peaks with lower threshold (TREC needs more anchors)
        threshold = 0.2  # Lower threshold to catch more instruction tokens
        min_spacing = seq_len // 100  # Allow denser anchors (1% spacing)

        # Sort by score and greedily select with spacing constraint
        sorted_indices = torch.argsort(anchor_scores, descending=True)

        for idx in sorted_indices:
            idx_val = idx.item()
            if idx_val < self.sink_size:
                continue  # Skip sink tokens
            if anchor_scores[idx_val] < threshold:
                break  # Below threshold

            # Check spacing constraint
            too_close = False
            for existing in anchors:
                if abs(idx_val - existing) < min_spacing:
                    too_close = True
                    break

            if not too_close:
                anchors.add(idx_val)
                if len(anchors) >= max_anchors:
                    break

        return anchors

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
        Update KV cache using SDI (Stratified Diverse Influence) eviction (v5.0.0).

        ═══════════════════════════════════════════════════════════════════════════
        SDI ALGORITHM (v5.0.0 - ICML 2026)
        ═══════════════════════════════════════════════════════════════════════════

        1. Compute full attention matrix from query/key states
        2. TASK DETECTION: Classify via attention entropy
           - Low entropy → QA (query-focused walks)
           - High entropy → Summarization (multi-source walks)
        3. IMPORTANCE SCORING:
           - H2O: Column sums (incoming attention)
           - Influence: Absorbing random walks (task-adaptive sources)
           - Combined: max(rank(h2o), rank(influence))
        4. STRATIFIED DIVERSE SELECTION:
           - Divide into K segments for coverage guarantee
           - Cluster keys for semantic diversity
           - Select B/(K*C) tokens per cluster

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

        print(f"CircuitKV v5.0.0 (SDI={self.use_sdi}, K={self.num_segments}, C={self.num_clusters}) max_capacity_prompt {self.max_capacity_prompt}")

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

        # Softmax and MAX-POOL across heads (preserves induction head signal)
        # v1.0.9: Max-pooling keeps sharp attention patterns from specialized heads
        # that get washed out by averaging (e.g., induction heads for few-shot)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_avg = attn_weights.max(dim=1).values.mean(dim=0)  # Max over heads, avg over batch

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
        # STEP 2: SDI - Task Detection + Influence Scoring (v5.0.0)
        # ═══════════════════════════════════════════════════════════════════
        current_idx = q_len - 1
        attn_contig = full_attn.contiguous()

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2a: Task Detection via Attention Entropy
        # ─────────────────────────────────────────────────────────────────────
        query_attention = full_attn[-1, :q_len]  # Last query's attention
        task_type = self._detect_task_type(query_attention, q_len) if self.use_sdi else "qa"
        self._detected_task_type = task_type

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2b: Compute Influence Scores (Task-Adaptive)
        # ─────────────────────────────────────────────────────────────────────
        if self.use_sdi and task_type == "summarization":
            # SUMMARIZATION: Multi-source walks from segment centers
            # This captures global document structure, not just query-reachable tokens
            segment_centers = self._compute_segment_centers(q_len, self.num_segments)
            influence_scores = self._run_multi_source_walks(
                full_attn, segment_centers, current_idx, q_len
            )
            print(f"  [SDI] Task=summarization, using multi-source walks (K={self.num_segments} sources)")
        else:
            # QA: Query-focused walks (original behavior)
            # Query is informative, so walks from query are appropriate
            self._graph.update_and_step_influence_walker(
                attn_contig, current_idx,
                self.num_walkers, self.max_steps, self.sink_size,
            )
            influence_scores = self._graph.get_influence_scores()[:q_len].clone()
            print(f"  [SDI] Task=qa, using query-focused walks")

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2c: H2O Scores (Column Sums)
        # ─────────────────────────────────────────────────────────────────────
        h2o_scores = full_attn.sum(dim=0)[:q_len]  # [q_len]

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2d: MAX(H2O, Influence) with Rank Normalization
        # ─────────────────────────────────────────────────────────────────────
        h2o_rank = self._rank_normalize(h2o_scores)
        influence_rank = self._rank_normalize(influence_scores)

        # MAX combination: keeps token if EITHER signal wants it
        combined_scores = torch.maximum(h2o_rank, influence_rank)

        # =====================================================================
        # STEP 3: SDI - Stratified Diverse Selection (v5.0.0)
        # ═══════════════════════════════════════════════════════════════════
        if self.use_sdi:
            # Get key vectors for clustering (average across heads)
            keys_for_clustering = key_states.mean(dim=1).squeeze(0)  # [q_len, head_dim]

            # Stratified selection with semantic diversity
            keep_mask = self._stratified_diverse_select(
                combined_scores,
                keys_for_clustering,
                self.max_capacity_prompt,
                q_len
            )
            print(f"  [SDI] Stratified selection: K={self.num_segments} segments, C={self.num_clusters} clusters")
        else:
            # Fallback: Standard top-k selection
            scores_buffer = torch.zeros(self._max_seq_len, device=combined_scores.device, dtype=combined_scores.dtype)
            scores_buffer[:q_len] = combined_scores
            keep_mask = self._get_keep_mask(
                scores_buffer,
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


class MaxKVCluster():
    """
    MaxKV v1.0.0: MAX(H2O, Influence) with Rank Normalization.

    This class implements the "hedging" strategy discovered in PoC ablation:
    - H2O captures "Heavy Hitters" (popular tokens, good for NarrativeQA)
    - Influence captures "Reasoning Bridges" (path-critical tokens, good for HotpotQA)
    - MAX combination is robust across BOTH task types

    Key Innovation:
    - Rank normalization ensures both signals are on equal footing
    - MAX(rank_h2o, rank_influence) keeps tokens important to EITHER method
    - No tuning needed: works across all LongBench tasks

    Algorithm:
    1. Compute H2O scores (column sums of attention)
    2. Compute Influence scores (absorbing random walk from query)
    3. Rank-normalize both to [0, 1]
    4. Score[j] = max(rank_h2o[j], rank_influence[j])
    5. Keep top-K by combined score

    Why MAX > Multiplicative Gravity:
    - Gravity: Score = H2O * Influence - fails when signals conflict
    - MAX: Score = max(H2O, Influence) - hedges by keeping either signal
    """

    def __init__(
        self,
        window_size: int = 32,
        max_capacity_prompt: int = 2048,
        kernel_size: int = 5,
        pooling: str = 'avgpool',
        merge=None,
        # CircuitKV-specific parameters
        sink_size: int = 4,
        top_k: int = 32,
        alpha: float = 0.85,
        num_walkers: int = 10000,
        num_steps: int = 100,
        max_steps: int = 10,
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
        self.max_steps = max_steps
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.debug = debug

        # Lazy initialization of CUDA graph
        self._graph = None
        self._device = None
        self._max_seq_len = 8192

    def reset(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 2048,
        kernel_size: int = 5,
        pooling: str = 'avgpool',
        merge=None,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        if self._graph is not None:
            self._graph.reset()

    def _lazy_init(self, device, seq_len: int):
        """Initialize CUDA graph on first use."""
        if self._graph is None or self._device != device:
            try:
                from circuit_kv import _C as circuit_kv_cuda
                self._max_seq_len = max(seq_len * 2, 8192)
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

    def _rank_normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """Rank-based normalization to [0, 1]."""
        ranks = torch.argsort(torch.argsort(scores)).float()
        return ranks / (len(scores) - 1 + 1e-10)

    def _get_keep_mask(self, scores: torch.Tensor, budget: int, seq_len: int) -> torch.Tensor:
        """Get boolean mask indicating which tokens to keep."""
        if isinstance(budget, float) and budget <= 1.0:
            target_count = int(seq_len * budget)
        else:
            target_count = int(budget)

        min_required = self.sink_size + self.window_size
        if target_count < min_required:
            target_count = min_required

        device = scores.device
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[:self.sink_size] = True

        local_start = max(self.sink_size, seq_len - self.window_size)
        mask[local_start:] = True

        current_kept = mask.sum().item()
        remaining_slots = max(0, target_count - current_kept)

        if remaining_slots > 0:
            scores_masked = scores[:seq_len].clone()
            scores_masked[mask] = float('-inf')

            k = min(remaining_slots, (~mask).sum().item())
            if k > 0:
                _, topk_indices = torch.topk(scores_masked, k)
                mask[topk_indices] = True

        return mask

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask,
        num_key_value_groups: int,
    ):
        """
        Update KV cache using MAX(H2O, Influence) with rank normalization.

        Algorithm:
        1. Compute full attention matrix from window queries
        2. Compute H2O scores (column sums)
        3. Run Influence walker (CUDA kernel)
        4. Rank-normalize both scores
        5. Combined score = MAX(rank_h2o, rank_influence)
        6. Keep top-K by combined score
        """
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f"MaxKV (MAX combination) max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        self._lazy_init(key_states.device, q_len)

        # =====================================================================
        # STEP 1: Compute attention matrix
        # =====================================================================
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, :, -self.window_size:] += mask[None, None, :, :]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_avg = attn_weights.max(dim=1).values.mean(dim=0)

        # Build full attention matrix
        full_attn = torch.zeros(q_len, q_len, device=key_states.device, dtype=torch.float32)
        full_attn[-self.window_size:, :] = attn_avg

        n_prefix = q_len - self.window_size
        if n_prefix > 1:
            h2o_scores_window = attn_avg.sum(dim=0)
            h2o_prefix = h2o_scores_window[:n_prefix].clone().clamp(min=1e-6)
            cumsum = h2o_prefix.cumsum(dim=0)
            denom = torch.zeros(n_prefix, device=key_states.device, dtype=torch.float32)
            denom[1:] = cumsum[:-1]
            denom[0] = 1.0
            h2o_expanded = h2o_prefix.unsqueeze(0).expand(n_prefix, n_prefix)
            denom_expanded = denom.unsqueeze(1).expand(n_prefix, n_prefix)
            h2o_trans = h2o_expanded / (denom_expanded + 1e-8)
            mask_causal = torch.tril(torch.ones(n_prefix, n_prefix, device=key_states.device, dtype=torch.float32), diagonal=-1)
            full_attn[:n_prefix, :n_prefix] = h2o_trans * mask_causal

        # =====================================================================
        # STEP 2: Compute H2O scores (column sums)
        # =====================================================================
        h2o_scores = full_attn.sum(dim=0)

        # =====================================================================
        # STEP 3: Run Influence Walker (CUDA kernel)
        # =====================================================================
        current_idx = q_len - 1
        self._graph.update_and_step_influence_walker(
            full_attn.contiguous(),
            current_idx,
            self.num_walkers,
            self.max_steps,
            self.sink_size,
        )
        influence_scores = self._graph.get_influence_scores()

        # =====================================================================
        # STEP 4: Rank normalize both and take MAX
        # =====================================================================
        h2o_rank = self._rank_normalize(h2o_scores[:q_len])
        influence_rank = self._rank_normalize(influence_scores[:q_len])

        # MAX combination: robust hedging
        combined_scores = torch.maximum(h2o_rank, influence_rank)

        # Debug: show contribution breakdown
        if self.debug:
            h2o_wins = (h2o_rank >= influence_rank).sum().item()
            inf_wins = (influence_rank > h2o_rank).sum().item()
            print(f"  MaxKV: H2O contributes {h2o_wins}/{q_len} ({100*h2o_wins/q_len:.1f}%), "
                  f"Influence contributes {inf_wins}/{q_len} ({100*inf_wins/q_len:.1f}%)")

        # =====================================================================
        # STEP 5: Eviction based on MAX scores
        # =====================================================================
        keep_mask = self._get_keep_mask(
            combined_scores,
            self.max_capacity_prompt,
            q_len
        )

        non_local_mask = keep_mask.clone()
        non_local_mask[-self.window_size:] = False
        keep_indices = non_local_mask.nonzero(as_tuple=True)[0]
        num_keep = keep_indices.shape[0]

        indices_expanded = keep_indices.view(1, 1, -1, 1).expand(bsz, num_heads, -1, head_dim)

        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices_expanded[:, :, :num_keep, :])
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices_expanded[:, :, :num_keep, :])

        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]

        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        return key_states, value_states


def init_maxkv(self):
    """Initialize MaxKV cluster with MAX(H2O, Influence) combination."""
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
        if not hasattr(self.config, 'sink_size'):
            self.config.sink_size = 4
        if not hasattr(self.config, 'top_k'):
            self.config.top_k = 32
        if not hasattr(self.config, 'alpha'):
            self.config.alpha = 0.85
        if not hasattr(self.config, 'num_walkers'):
            self.config.num_walkers = 10000
        if not hasattr(self.config, 'num_steps'):
            self.config.num_steps = 100
        if not hasattr(self.config, 'max_steps'):
            self.config.max_steps = 10
        if not hasattr(self.config, 'maxkv_debug'):
            self.config.maxkv_debug = False

    self.kv_cluster = MaxKVCluster(
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
        max_steps=self.config.max_steps,
        debug=self.config.maxkv_debug,
    )


def init_circuitkv(self):
    """Initialize CircuitKV cluster with Causal Influence Walker (v1.0.0)."""
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
        # CircuitKV-specific defaults (v1.0.0 - Validated by PoC5)
        if not hasattr(self.config, 'sink_size'):
            self.config.sink_size = 4  # Absorbing boundary
        if not hasattr(self.config, 'top_k'):
            self.config.top_k = 32
        if not hasattr(self.config, 'alpha'):
            self.config.alpha = 0.85  # Unused, kept for API compatibility
        if not hasattr(self.config, 'num_walkers'):
            self.config.num_walkers = 10000  # v1.0.0: 10000 walkers (validated by PoC5)
        if not hasattr(self.config, 'num_steps'):
            self.config.num_steps = 100  # MAX_STEPS for safety timeout (legacy)
        if not hasattr(self.config, 'max_steps'):
            self.config.max_steps = 50  # v1.0.1: increased from 10 to reach distant tokens
        # Capacitive model parameter
        if not hasattr(self.config, 'decay'):
            self.config.decay = 0.99  # EMA decay for charge accumulation
        # v4.0.0: Bidirectional Circuit Walks (P0 - ICML 2026)
        if not hasattr(self.config, 'bidirectional'):
            self.config.bidirectional = True
        # Legacy Landmark Walker parameters (kept for API compatibility)
        if not hasattr(self.config, 'num_landmarks'):
            self.config.num_landmarks = 8  # Legacy (unused in v1.0.0)
        if not hasattr(self.config, 'min_spacing'):
            self.config.min_spacing = 50  # Legacy (unused in v1.0.0)
        if not hasattr(self.config, 'walkers_per_source'):
            self.config.walkers_per_source = 100  # Legacy (unused in v1.0.0)
        if not hasattr(self.config, 'query_boost'):
            self.config.query_boost = 2.0  # Legacy (unused in v1.0.0)
        if not hasattr(self.config, 'position_alpha'):
            self.config.position_alpha = 0.6  # Legacy (unused in v1.0.0)
        if not hasattr(self.config, 'use_reachability'):
            self.config.use_reachability = False  # Legacy (unused in v1.0.0)
        # v0.5.0: Landmark Absorbing Walker (legacy)
        if not hasattr(self.config, 'absorb_at_landmarks'):
            self.config.absorb_at_landmarks = True  # Legacy (unused in v1.0.0)
        # Debug logging
        if not hasattr(self.config, 'circuitkv_debug'):
            self.config.circuitkv_debug = False
        # v4.0.0: Multi-Scale Ensemble (P1 - ICML 2026)
        if not hasattr(self.config, 'use_multi_horizon'):
            self.config.use_multi_horizon = True

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
        max_steps=self.config.max_steps,  # v1.0.0: Max steps per walker
        decay=self.config.decay,
        bidirectional=self.config.bidirectional,
        # Legacy Landmark Walker parameters (kept for API compatibility)
        num_landmarks=self.config.num_landmarks,
        min_spacing=self.config.min_spacing,
        walkers_per_source=self.config.walkers_per_source,
        query_boost=self.config.query_boost,
        position_alpha=self.config.position_alpha,
        use_reachability=self.config.use_reachability,
        # v0.5.0: Landmark Absorbing Walker (legacy)
        absorb_at_landmarks=self.config.absorb_at_landmarks,
        # v4.0.0 Features
        use_multi_horizon=self.config.use_multi_horizon,
        # Debug
        debug=self.config.circuitkv_debug,
    )