import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import os

from typing import List


from typing import List, Optional, Tuple, Set
from transformers.cache_utils import Cache

# v3.0.0 Breakthroughs import
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

# Layer Fairness Diagnostic - track per-layer stats to detect "Participation Trophy Effect"
_CIRCUITKV_LAYER_STATS = []  # List of (layer_idx, kept_count, raw_max_qi, raw_max_hi, raw_max_combined)

def _set_circuitkv_dataset(dataset_name: str):
    """Set the current dataset name for debug logging."""
    global _CIRCUITKV_CURRENT_DATASET
    _CIRCUITKV_CURRENT_DATASET = dataset_name

def _get_circuitkv_debug_log():
    """Get or create the shared debug log file (CSV format)."""
    global _CIRCUITKV_DEBUG_LOG, _CIRCUITKV_DEBUG_INITIALIZED
    if not _CIRCUITKV_DEBUG_INITIALIZED:
        log_path = os.path.join(os.getcwd(), "circuitkv_debug.csv")
        _CIRCUITKV_DEBUG_LOG = open(log_path, "w")
        # CSV header for layer fairness data
        _CIRCUITKV_DEBUG_LOG.write("sample,dataset,layer,kept,raw_qi,raw_hi\n")
        _CIRCUITKV_DEBUG_INITIALIZED = True
    return _CIRCUITKV_DEBUG_LOG

def _circuitkv_debug_next_sample():
    """Increment sample counter, dump layer fairness, and reset layer counter."""
    global _CIRCUITKV_SAMPLE_COUNTER, _CIRCUITKV_LAYER_COUNTER

    # Dump previous sample's layer fairness stats before starting new sample
    if _CIRCUITKV_LAYER_STATS:
        _dump_layer_fairness()

    _CIRCUITKV_SAMPLE_COUNTER += 1
    _CIRCUITKV_LAYER_COUNTER = 0
    return _CIRCUITKV_SAMPLE_COUNTER

def _circuitkv_debug_next_layer():
    """Increment layer counter."""
    global _CIRCUITKV_LAYER_COUNTER
    _CIRCUITKV_LAYER_COUNTER += 1
    return _CIRCUITKV_LAYER_COUNTER

def _record_layer_fairness(layer_idx: int, kept_count: int, raw_max_qi: float, raw_max_hi: float, raw_max_combined: float):
    """Record per-layer stats for Layer Fairness diagnostic."""
    global _CIRCUITKV_LAYER_STATS
    _CIRCUITKV_LAYER_STATS.append((layer_idx, kept_count, raw_max_qi, raw_max_hi, raw_max_combined))

def _dump_layer_fairness():
    """Dump layer fairness stats in CSV format (one row per layer)."""
    global _CIRCUITKV_LAYER_STATS
    if not _CIRCUITKV_LAYER_STATS:
        return

    log = _get_circuitkv_debug_log()
    dataset = _CIRCUITKV_CURRENT_DATASET or "unknown"
    sample_num = _CIRCUITKV_SAMPLE_COUNTER

    # Write one CSV row per layer
    for layer_idx, kept, raw_qi, raw_hi, _ in sorted(_CIRCUITKV_LAYER_STATS):
        log.write(f"{sample_num},{dataset},{layer_idx},{kept},{raw_qi:.6f},{raw_hi:.6f}\n")

    log.flush()
    _CIRCUITKV_LAYER_STATS = []

def _reset_layer_fairness():
    """Reset layer fairness stats for a new sample."""
    global _CIRCUITKV_LAYER_STATS
    _CIRCUITKV_LAYER_STATS = []

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

    def _chunked_attention_fallback(self, key_states, query_states, value_states, head_dim, chunk_size=4096):
        """Fallback to chunked attention when full attention causes OOM."""
        bsz, num_heads, q_len, _ = query_states.shape
        past_len = q_len - self.window_size
        num_keep = self.max_capacity_prompt - self.window_size

        # Accumulator for attention scores
        attn_scores = torch.zeros(
            bsz, num_heads, past_len,
            device=key_states.device,
            dtype=torch.float32
        )

        past_keys = key_states[:, :, :past_len, :]

        # Process queries in chunks
        for i in range(0, q_len, chunk_size):
            end_i = min(i + chunk_size, q_len)
            q_chunk = query_states[:, :, i:end_i, :]

            attn_chunk = torch.matmul(q_chunk, past_keys.transpose(2, 3)) / math.sqrt(head_dim)
            attn_chunk = nn.functional.softmax(attn_chunk, dim=-1, dtype=torch.float32)
            attn_scores += attn_chunk.sum(dim=2)

            del attn_chunk
            torch.cuda.empty_cache()

        indices = attn_scores.topk(num_keep, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        return indices, past_len

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f"H2O max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            past_len = q_len - self.window_size

            try:
                # Try full attention first (faster, but requires more VRAM)
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

                # Apply causal mask
                mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                causal_mask = mask[None, None, :, :]
                attn_weights = attn_weights + causal_mask

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim=-2)

                indices = attn_weights_sum.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

                del attn_weights, attn_weights_sum, mask, causal_mask

            except torch.cuda.OutOfMemoryError:
                # Fallback to chunked attention on OOM
                print(f"H2O: CUDA OOM at seq_len={q_len}, falling back to chunked attention...")
                torch.cuda.empty_cache()
                indices, past_len = self._chunked_attention_fallback(key_states, query_states, value_states, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :past_len, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :past_len, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
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
    CircuitKV v4.3.0: Unified Dual-Importance via Absorbing Markov Chains.

    This class implements importance scoring for KV cache eviction using
    the FUNDAMENTAL MATRIX of absorbing Markov chains.

    v4.3.0 Key Innovation - UNIFIED IMPORTANCE SCORING:
    Both importance signals derived from the SAME mathematical object (N):

    1. QUERY IMPORTANCE (QI): N[q, j] = expected visits FROM query to j
       - "How important is j for answering THIS specific query?"
       - Captures tokens on paths from query to sink

    2. HUB IMPORTANCE (HI): (1/n) Σᵢ N[i, j] = average expected visits to j
       - "How central is j in the overall information flow network?"
       - Captures globally important "hub" tokens (multi-hop, not one-hop H2O)

    Final Score: MAX(QI_rank, HI_rank)
    - Keeps token if EITHER query-relevant OR globally central
    - "OR logic" is more robust than geometric mean's "AND logic"
    - Both signals from same theory (cleaner than mixing H2O with QI)

    Evolution:
    - v4.0: MAX(H2O, QI) - H2O from A (one-hop), QI from N (multi-hop)
    - v4.2: √(QI × HI) - both from N, but geometric mean too selective
    - v4.3: MAX(HI, QI) - both from N, with robust OR logic ← CURRENT

    Mathematical Foundation:
    - Attention matrix A defines transition probabilities P[i,j]
    - Q = transition matrix among transient states (non-sink tokens)
    - N = (I - Q)^{-1} = fundamental matrix of absorbing chain
    - Computed via Neumann series: I + Q + Q² + ... + Q^k

    Combination Modes (combination_mode parameter):
    - "dis": MAX(HI, QI) - v4.3.0, both from fundamental matrix N
    - "max": MAX(H2O, QI) - v4.0.0, H2O from attention, QI from N
    - "weighted": α * H2O + (1-α) * QI - v4.1.0

    Why v4.3 (dis) > v4.0 (max):
    - v4.0 mixes one-hop (H2O) with multi-hop (QI) - different theories
    - v4.3 uses multi-hop for BOTH signals - unified theory
    - HI captures bridge tokens that H2O misses

    Configuration:
    - combination_mode: "dis" (recommended), "max", or "weighted"
    - neumann_iterations: Neumann series iterations (default: 10)
    - neumann_temperature: Attention sharpening (default: 1.0, lower = sharper)
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
        # RC+B: Bidirectional Circuit Walks
        # NOTE: Disabled - R@65 improved but narrativeqa dropped (25.10 -> 23.78)
        bidirectional: bool = False,
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
        # v3.0.0 Breakthroughs (ICML 2026)
        use_instruction_anchors: bool = False,  # Breakthrough 2: DISABLED (TREC issue was prompting, not KV)
        use_fundamental_norm: bool = False,  # Breakthrough 1: Principled normalization (expensive)
        use_multi_horizon: bool = True,  # Breakthrough 3: Adaptive walk lengths
        tokenizer = None,  # Required for instruction anchor detection
        # v4.0.0: Deterministic Neumann Series
        use_neumann: bool = True,  # v4.0.0: Use deterministic Neumann instead of random walks
        neumann_iterations: int = 10,  # Number of Neumann series iterations
        neumann_temperature: float = 1.0,  # Temperature for attention sharpening (lower = sharper)
        neumann_gamma: float = 1.0,  # v6.11.0: Spectral decay factor (1.0=no decay, <1=locality bias)
        # v4.1.0: Combination tuning
        h2o_weight: float = 0.5,  # Weight for H2O in combination (0.5 = equal, >0.5 = favor H2O)
        combination_mode: str = "dis",  # "dis" (default, no DA), "max", "weighted", "union", "union_da"
        # v4.2.0: Dual-Importance Scoring
        dis_alpha: float = 0.5,  # QI weight in DIS (0.5 = symmetric geometric mean)
        # v5.0.0: Union Selection
        qi_ratio: float = 0.5,  # Ratio of budget for QI in Union mode (0.5 = 50% QI, 50% HI)
        # v4.3.1: Ablation flags for A1 experiment
        ablate_qi: bool = False,  # If True, use HI only (QI zeroed)
        ablate_hi: bool = False,  # If True, use QI only (HI zeroed)
        # v7.0.0: Per-head Markov importance (principled per-head selection)
        per_head_eviction: bool = False,  # If True, each head computes its own QI/HI
        head_chunk_size: int = 8,  # Number of heads to process in parallel (memory vs speed)
        # v6.1.0: Smoothing Kernel for phrase preservation
        smoothing_kernel: int = 0,  # Kernel size for score smoothing (0=disabled, 5=recommended)
        # v6.2.0: Asymmetric Gaussian Smoothing
        smooth_hi_only: bool = False,  # If True, only smooth HI (keep QI sharp for precision)
        use_gaussian: bool = False,  # If True, use Gaussian kernel instead of boxcar
        qi_kernel_size: int = 0,  # Separate kernel size for QI (0=use smoothing_kernel, -1=raw)
        hi_kernel_size: int = 0,  # Separate kernel size for HI (0=use smoothing_kernel)
        gaussian_sigma: float = 1.0,  # Sigma for Gaussian kernel
        # v6.5.0: Entropy-Aware Head Selection
        entropy_aware: bool = False,  # If True, use sharpest heads for QI, all heads for HI
        top_k_heads: int = 8,  # Number of sharpest (lowest entropy) heads for QI
        # v6.5.1: Head Selection Mode - choose between entropy and transient mass
        head_selection_mode: str = "entropy",  # "entropy" (lowest entropy) or "mass" (highest transient mass)
        # v6.7.0: HI Pooling Mode - how to aggregate heads for HI
        hi_pooling_mode: str = "mean",  # "mean" (consensus, v6.7) or "max" (peak, v6.5)
        # v6.8.0: Mass-Filtered Hubs - prune dead heads from HI consensus
        hi_mass_threshold: float = 0.0,  # Min transient mass to include head in HI (0=all, 0.1=filter dead)
        # v6.8.1: Top-K Mass Heads for HI - select top-k heads by transient mass instead of threshold
        hi_top_k_heads: int = 0,  # 0=disabled (use threshold), >0=select top-k heads by mass for HI
        # v6.9.0: CDF-Based Head Selection - select heads until cumulative mass >= threshold
        hi_mass_cdf: float = 0.0,  # 0=disabled, 0.85=select heads covering 85% of total transient mass
        # Diagnostic: log head selection stats
        hi_log_head_stats: bool = False,
        # v6.10.0: Bridge Importance (BI) via A² - captures 2-hop attention paths for cross-document reasoning
        use_bridge_importance: bool = False,  # Enable A² bridge importance for multi-hop
        bi_kernel_size: int = 5,  # Smoothing kernel for BI (bridges need context)
        # v6.12.0: HI Signal Gating - silence noisy HI in deep layers
        hi_signal_threshold: float = 0.0,  # If max(hi_raw) < threshold, zero out hi_rank (0=disabled)
        # v6.12.1: HI Soft Scaling - scale rank by raw max (automatic dampening)
        hi_scale_by_max: bool = False,  # If True, hi_rank = rank_normalize(hi) * hi_raw_max
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.sink_size = sink_size
        self.top_k = top_k
        self.alpha = alpha
        self.num_walkers = num_walkers
        self.num_steps = num_steps
        self.max_steps = max_steps  # v1.0.0: Max steps per walker
        # v3.0.0 Breakthroughs
        self.use_instruction_anchors = use_instruction_anchors
        self.use_fundamental_norm = use_fundamental_norm
        self.use_multi_horizon = use_multi_horizon
        self.tokenizer = tokenizer
        self._instruction_anchors = set()  # Cache for current sample
        # v4.0.0: Deterministic Neumann Series
        self.use_neumann = use_neumann
        self.neumann_iterations = neumann_iterations
        self.neumann_temperature = neumann_temperature
        self.neumann_gamma = neumann_gamma
        # v4.1.0: Combination tuning
        self.h2o_weight = h2o_weight
        self.combination_mode = combination_mode
        # v4.2.0: Dual-Importance Scoring
        self.dis_alpha = dis_alpha
        # v5.0.0: Union Selection
        self.qi_ratio = qi_ratio
        # v4.3.1: Ablation flags
        self.ablate_qi = ablate_qi
        self.ablate_hi = ablate_hi
        # v7.0.0: Per-head Markov importance
        self.per_head_eviction = per_head_eviction
        self.head_chunk_size = head_chunk_size
        # v6.1.0: Smoothing Kernel
        self.smoothing_kernel = smoothing_kernel
        # v6.2.0: Asymmetric Gaussian Smoothing
        self.smooth_hi_only = smooth_hi_only
        self.use_gaussian = use_gaussian
        self.qi_kernel_size = qi_kernel_size
        self.hi_kernel_size = hi_kernel_size
        self.gaussian_sigma = gaussian_sigma
        # v6.5.0: Entropy-Aware Head Selection
        self.entropy_aware = entropy_aware
        self.top_k_heads = top_k_heads
        # v6.5.1: Head Selection Mode
        self.head_selection_mode = head_selection_mode
        # v6.7.0: HI Pooling Mode
        self.hi_pooling_mode = hi_pooling_mode
        # v6.8.0: Mass-Filtered Hubs
        self.hi_mass_threshold = hi_mass_threshold
        # v6.8.1: Top-K Mass Heads for HI
        self.hi_top_k_heads = hi_top_k_heads
        # v6.9.0: CDF-Based Head Selection
        self.hi_mass_cdf = hi_mass_cdf
        # Diagnostic logging
        self.hi_log_head_stats = hi_log_head_stats
        # v6.10.0: Bridge Importance
        self.use_bridge_importance = use_bridge_importance
        self.bi_kernel_size = bi_kernel_size
        # v6.12.0: HI Signal Gating
        self.hi_signal_threshold = hi_signal_threshold
        # v6.12.1: HI Soft Scaling
        self.hi_scale_by_max = hi_scale_by_max
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
        self._version_printed = False  # Only print version info once

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

    def _compute_head_entropy(self, attn_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        v6.5.0: Compute entropy per attention head.

        Sharp heads (low entropy) have focused attention on specific tokens,
        which is characteristic of induction heads that carry multi-hop reasoning.

        Args:
            attn_weights: [bsz, num_heads, window_size, seq_len] - softmax attention
            eps: Small value to avoid log(0)

        Returns:
            entropy: [num_heads] - entropy per head (lower = sharper)
        """
        # CRITICAL: Convert to float32 to avoid NaN from float16 log operations
        attn_f32 = attn_weights.float()
        attn_clamped = attn_f32.clamp(min=eps)
        log_attn = torch.log(attn_clamped)

        # H = -sum(p * log(p)) for each query position
        entropy_per_query = -(attn_clamped * log_attn).sum(dim=-1)  # [bsz, num_heads, window_size]

        # Average entropy across batch and query positions
        entropy_per_head = entropy_per_query.mean(dim=(0, 2))  # [num_heads]

        return entropy_per_head

    def _compute_head_transient_mass(self, attn_weights: torch.Tensor, sink_size: int) -> torch.Tensor:
        """
        v6.5.1: Compute transient mass per attention head.

        Transient mass = sum of attention to NON-SINK tokens.
        High transient mass means the head connects content to content.
        Low transient mass means the head mostly looks at sink tokens.

        This is a more direct measure than entropy because:
        - A sharp head (low entropy) could attend 100% to sink → useless for reasoning
        - A high transient mass head actually connects content tokens

        Args:
            attn_weights: [bsz, num_heads, window_size, seq_len] - softmax attention
            sink_size: Number of sink tokens at the beginning

        Returns:
            transient_mass: [num_heads] - transient mass per head (higher = more content connectivity)
        """
        # Sum attention to non-sink tokens (transient states in Markov chain terminology)
        # attn_weights[..., sink_size:] = attention to content tokens only
        transient_attn = attn_weights[..., sink_size:]  # [bsz, heads, window, seq-sink]

        # Sum across the key dimension to get row sums (how much each query looks at content)
        row_sums = transient_attn.sum(dim=-1)  # [bsz, heads, window]

        # Average across batch and query positions to get per-head score
        transient_mass_per_head = row_sums.mean(dim=(0, 2))  # [num_heads]

        return transient_mass_per_head

    def _compute_bridge_importance(self, attn_matrix: torch.Tensor) -> torch.Tensor:
        """
        v6.10.0: Compute Bridge Importance via A² (second-order attention).

        For multi-hop reasoning (especially cross-document like 2WikiMQA), we need to
        capture tokens that are important through 2-hop paths:

            Query → Bridge Token → Target Token

        In Markov chain terms:
        - A = one-step transition probabilities
        - A² = two-step transition probabilities

        Bridge Importance captures tokens that are reachable in 2 hops, which identifies
        "bridge" tokens that connect different parts of the context (e.g., entities that
        appear in both documents in 2WikiMQA).

        Args:
            attn_matrix: [seq_len, seq_len] - full attention matrix (already aggregated across heads)

        Returns:
            bi_scores: [seq_len] - bridge importance scores per token
        """
        # Compute A² = A @ A (two-hop attention)
        # A²[i,j] = sum_k A[i,k] * A[k,j] = probability of reaching j from i in 2 steps
        A_squared = torch.matmul(attn_matrix, attn_matrix)  # [seq, seq]

        # Bridge Importance: How much is each token reachable in 2 hops from any query?
        # Sum over all query positions (rows) to get total 2-hop reachability
        bi_scores = A_squared.sum(dim=0)  # [seq_len]

        # Alternative: Focus on 2-hop reach from recent queries (observation window)
        # This emphasizes bridges relevant to the current question
        # window_size = min(32, attn_matrix.shape[0])
        # bi_scores = A_squared[-window_size:, :].sum(dim=0)  # [seq_len]

        return bi_scores

    def _init_debug_log(self):
        """Initialize debug log (uses module-level singleton)."""
        log = _get_circuitkv_debug_log()

    def _log_debug(
        self,
        q_len: int,
        h2o_scores: torch.Tensor,
        combined_scores: torch.Tensor,
        keep_mask: torch.Tensor,
        full_attn: torch.Tensor,
        qi_scores: torch.Tensor = None,
        hi_scores: torch.Tensor = None,
        attn_weights_per_head: torch.Tensor = None,
        layer_idx: int = 1,
    ):
        """Debug logging helper. Sample transition handled externally."""
        if not self.debug:
            return
        # Layer data already recorded via _record_layer_fairness
        # Sample transition is now triggered from run_longbench.py

    def _lazy_init(self, device, seq_len: int):
        """Initialize CUDA graph on first use (only needed for random walk mode)."""
        # v6.5.0: Skip CUDA graph init when using Neumann series (pure PyTorch path)
        if self.use_neumann:
            self._device = device
            return

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

    def _apply_smoothing(self, scores: torch.Tensor, kernel_size: int, use_gaussian: bool = False, sigma: float = 1.0) -> torch.Tensor:
        """
        v6.2.0: Apply 1D smoothing kernel to preserve phrase structure.

        The "Token Isolation" Problem:
        - Raw Markov scores spike on individual important tokens
        - Selection keeps "Paris" but deletes "is" from "Paris is the capital"
        - LLM loses syntactic context needed for reasoning

        Solution: Low-pass filter (1D convolution) that promotes neighbors of important tokens.
        If token j is important, tokens j-1 and j+1 get boosted to "survival status."

        v6.2.0 Upgrade: Gaussian kernel option
        - Boxcar: [0.2, 0.2, 0.2, 0.2, 0.2] - flattens the spike (80% penalty to center)
        - Gaussian: [0.1, 0.2, 0.4, 0.2, 0.1] - center stays dominant, neighbors get lift
        Gaussian preserves PRECISION while adding COHERENCE.

        Args:
            scores: Raw importance scores [seq_len]
            kernel_size: Size of smoothing window (odd recommended: 3, 5, 7)
            use_gaussian: If True, use Gaussian kernel; if False, use boxcar (default)
            sigma: Standard deviation for Gaussian kernel (default 1.0)

        Returns:
            Smoothed scores [seq_len] preserving phrase structure
        """
        if kernel_size <= 1:
            return scores

        n = len(scores)
        if n <= kernel_size:
            return scores

        # Reshape for Conv1d: [batch=1, channel=1, length=n]
        scores_3d = scores.view(1, 1, -1)

        if use_gaussian:
            # v6.2.0: Gaussian kernel - preserves center spike while lifting neighbors
            # kernel[i] = exp(-(i - center)^2 / (2*sigma^2))
            x = torch.arange(kernel_size, device=scores.device, dtype=scores.dtype) - (kernel_size - 1) / 2
            weight = torch.exp(-x**2 / (2 * sigma**2))
            weight = weight / weight.sum()  # Normalize to sum to 1
            weight = weight.view(1, 1, -1)
        else:
            # v6.1.0: Boxcar (averaging) kernel
            weight = torch.ones(1, 1, kernel_size, device=scores.device, dtype=scores.dtype) / kernel_size

        # Padding to preserve length (phase-aligned output)
        padding = kernel_size // 2

        # Apply convolution
        smoothed = F.conv1d(scores_3d, weight, padding=padding)

        return smoothed.flatten()[:n]  # Ensure exact length match

    def _compute_influence_neumann(
        self,
        attention: torch.Tensor,
        query_idx: int,
        sink_size: int = 4,
        num_iterations: int = 10,
        temperature: float = 1.0,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """
        v4.0.0: Compute deterministic influence scores via Neumann series.
        v6.11.0: Added spectral decay factor gamma for locality bias.

        Computes expected visit counts analytically using the Fundamental Matrix
        of an absorbing Markov chain:

            N = (I - Q)^{-1} ≈ I + Q + Q² + ... + Q^k

        Where Q is the substochastic transition matrix among transient states
        (non-sink tokens). The influence score for token j is the expected
        number of times a walk from query visits j before absorption.

        Mathematical Properties:
        - Deterministic: No randomness, perfectly reproducible
        - Convergent: Neumann series converges since Q is substochastic
        - Efficient: O(k * n²) via iterative matrix-vector products

        Args:
            attention: Causal attention matrix [seq_len, seq_len], already masked
            query_idx: Source position (typically seq_len - 1)
            sink_size: First sink_size tokens are absorbing (default 4)
            num_iterations: Neumann series iterations (default 10)
            temperature: Attention sharpening (default 1.0, lower = sharper)

        Returns:
            Influence scores [seq_len] - expected visits from query
        """
        n = attention.shape[0]
        device = attention.device

        # Handle edge cases
        if n <= sink_size:
            return torch.ones(n, device=device, dtype=torch.float32) / n

        # =====================================================================
        # STEP 1: Build transition matrix P from attention
        # P[i,j] = probability of transitioning from i to j
        # =====================================================================
        # Apply temperature scaling for sharpness control
        if temperature != 1.0 and temperature > 0:
            # Sharpen or soften attention distribution
            attn_scaled = attention ** (1.0 / temperature)
        else:
            attn_scaled = attention

        # Row-normalize to get transition probabilities
        # Avoid division by zero for rows with no attention (sink rows)
        row_sums = attn_scaled.sum(dim=1, keepdim=True)
        row_sums = row_sums.clamp(min=1e-8)
        P = attn_scaled / row_sums  # [n, n]

        # =====================================================================
        # STEP 2: Extract Q (transient-to-transient transitions)
        # Transient states: indices >= sink_size
        # Q[i,j] = P[i + sink_size, j + sink_size]
        # =====================================================================
        n_transient = n - sink_size
        Q = P[sink_size:, sink_size:].contiguous()  # [n_transient, n_transient]

        # =====================================================================
        # STEP 3: Neumann series for query row of fundamental matrix
        # We only need N[query_idx, :] = sum_{k=0}^{inf} (Q^k)[query_idx, :]
        # Compute iteratively: v = e_q, result = e_q, then v = Q @ v, result += v
        # =====================================================================
        # Query position in transient space
        query_transient_idx = query_idx - sink_size
        if query_transient_idx < 0:
            # Query is in sink region - return uniform scores
            return torch.ones(n, device=device, dtype=torch.float32) / n

        # Initialize: e_query (one-hot at query position)
        v = torch.zeros(n_transient, device=device, dtype=torch.float32)
        v[query_transient_idx] = 1.0

        # Accumulator for Neumann series: starts with I (identity contribution)
        result = v.clone()

        # Iterate Neumann series: result = I + γQ + γ²Q² + ... + γ^k Q^k
        # v6.11.0: Added spectral decay factor gamma for locality bias
        # We want row q of N = (I-Q)^(-1), i.e., expected visits FROM query TO each j
        #
        # Row q of Q^k is computed as: (Q^k)[q,:] = e_q^T @ Q^k
        # In column form: ((Q^k)[q,:])^T = (Q^T)^k @ e_q
        #
        # So we iterate: v_{k+1} = γ * Q^T @ v_k, where v_0 = e_query
        # This gives v_k = γ^k * (Q^T)^k @ e_q = row q of γ^k Q^k (as column vector)
        for _ in range(num_iterations):
            v = gamma * torch.mv(Q.t(), v)  # γ * Q^T @ v - gives row q of γ^k Q^k
            result = result + v

            # Early stopping if converged
            if v.abs().max().item() < 1e-8:
                break

        # =====================================================================
        # STEP 4: Map back to full sequence and normalize
        # =====================================================================
        scores = torch.zeros(n, device=device, dtype=torch.float32)
        scores[sink_size:] = result

        # Sink tokens: assign small positive score (they absorb flow)
        # Option: could assign based on flow INTO sink
        scores[:sink_size] = result.sum() * 0.01  # Small fraction of total

        # Normalize to [0, 1] range (not rank normalization, just scale)
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        return scores

    def _compute_dual_importance_scores(
        self,
        attention: torch.Tensor,
        query_idx: int,
        sink_size: int = 4,
        num_iterations: int = 10,
        temperature: float = 1.0,
        attention_for_hi: torch.Tensor = None,
        gamma: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        v4.2.0: Dual-Importance Scoring (DIS) via Absorbing Markov Chain.
        v6.5.0: Added entropy-aware mode with separate attention for HI.
        v6.11.0: Added spectral decay factor gamma for locality bias.

        Computes TWO importance scores from the fundamental matrix N:

        1. QUERY IMPORTANCE (QI): N[q, j] = expected visits to j FROM query
           - "How important is j for answering THIS specific query?"
           - Captures query-relevant tokens on the path to sink
           - v6.5.0: Uses sharp-head attention (if entropy_aware)

        2. HUB IMPORTANCE (HI): (1/n) Σᵢ N[i, j] = avg expected visits to j
           - "How central is j in the overall information flow network?"
           - Captures globally important "hub" tokens
           - v6.5.0: Uses all-head attention (if entropy_aware)

        Mathematical Foundation:
        - N = (I - Q)^{-1} is the fundamental matrix of absorbing Markov chain
        - QI[j] = row q of N = expected visits from query position
        - HI[j] = column mean of N = average importance from ALL starting points

        Efficient Computation (O(k * n²) for BOTH metrics):
        - QI: Iterate v = Q.t() @ v starting from e_query
        - HI: Iterate u = Q.t() @ u starting from uniform (1/n)

        Why BOTH Matter:
        - QI alone misses globally important tokens that query doesn't directly see
        - HI alone misses query-specific relevant tokens
        - Together, they capture complementary notions of importance

        Combination Strategy (external):
        - DIS[j] = √(QI[j] × HI[j])  (geometric mean)
        - Requires BOTH properties - more selective than MAX

        Args:
            attention: Causal attention matrix [seq_len, seq_len] - used for QI
            query_idx: Source position (typically seq_len - 1)
            sink_size: First sink_size tokens are absorbing (default 4)
            num_iterations: Neumann series iterations (default 10)
            temperature: Attention sharpening (default 1.0, lower = sharper)
            attention_for_hi: Optional separate attention matrix for HI (v6.5.0 entropy-aware)

        Returns:
            Tuple of (query_importance, hub_importance), each [seq_len]
        """
        n = attention.shape[0]
        device = attention.device

        # Handle edge cases
        if n <= sink_size:
            uniform = torch.ones(n, device=device, dtype=torch.float32) / n
            return uniform, uniform

        # =====================================================================
        # STEP 1: Build transition matrix P from attention
        # =====================================================================
        if temperature != 1.0 and temperature > 0:
            attn_scaled = attention ** (1.0 / temperature)
        else:
            attn_scaled = attention

        row_sums = attn_scaled.sum(dim=1, keepdim=True).clamp(min=1e-8)
        P = attn_scaled / row_sums  # [n, n]

        # =====================================================================
        # STEP 2: Extract Q (transient-to-transient transitions)
        # =====================================================================
        n_transient = n - sink_size
        Q = P[sink_size:, sink_size:].contiguous()  # [n_transient, n_transient] - used for QI

        # v6.5.0: Build separate Q_hi for HI if entropy-aware attention provided
        if attention_for_hi is not None:
            if temperature != 1.0 and temperature > 0:
                attn_hi_scaled = attention_for_hi ** (1.0 / temperature)
            else:
                attn_hi_scaled = attention_for_hi
            row_sums_hi = attn_hi_scaled.sum(dim=1, keepdim=True).clamp(min=1e-8)
            P_hi = attn_hi_scaled / row_sums_hi
            Q_hi = P_hi[sink_size:, sink_size:].contiguous()
        else:
            Q_hi = Q  # Default: same Q for both

        # Query position in transient space
        query_transient_idx = query_idx - sink_size
        if query_transient_idx < 0:
            uniform = torch.ones(n, device=device, dtype=torch.float32) / n
            return uniform, uniform

        # =====================================================================
        # STEP 3: Parallel Neumann series for BOTH QI and HI
        # QI: Start from e_query (one-hot at query position) - uses Q (sharp heads)
        # HI: Start from uniform (1/n_transient) for average over all starts - uses Q_hi (all heads)
        # =====================================================================
        # Initialize Query Importance vector
        v_qi = torch.zeros(n_transient, device=device, dtype=torch.float32)
        v_qi[query_transient_idx] = 1.0
        result_qi = v_qi.clone()

        # Initialize Hub Importance vector (uniform start = average over all)
        v_hi = torch.ones(n_transient, device=device, dtype=torch.float32) / n_transient
        result_hi = v_hi.clone()

        # Iterate both in parallel (with potentially different Q matrices)
        # v6.11.0: Apply spectral decay factor gamma for locality bias
        # Computes: I + γQ + γ²Q² + ... + γ^k Q^k  (discounted multi-hop paths)
        for _ in range(num_iterations):
            v_qi = gamma * torch.mv(Q.t(), v_qi)
            v_hi = gamma * torch.mv(Q_hi.t(), v_hi)  # v6.5.0: Use Q_hi for HI
            result_qi = result_qi + v_qi
            result_hi = result_hi + v_hi

            # Early stopping if both converged
            if v_qi.abs().max().item() < 1e-8 and v_hi.abs().max().item() < 1e-8:
                break

        # =====================================================================
        # STEP 4: Map back to full sequence
        # =====================================================================
        qi_scores = torch.zeros(n, device=device, dtype=torch.float32)
        qi_scores[sink_size:] = result_qi
        qi_scores[:sink_size] = result_qi.sum() * 0.01  # Small score for sink

        hi_scores = torch.zeros(n, device=device, dtype=torch.float32)
        hi_scores[sink_size:] = result_hi
        hi_scores[:sink_size] = result_hi.sum() * 0.01  # Small score for sink

        # Capture raw max values BEFORE normalization (for Layer Fairness debug)
        qi_raw_max = qi_scores.max().item()
        hi_raw_max = hi_scores.max().item()

        # Normalize each to [0, 1]
        qi_max = qi_scores.max()
        if qi_max > 0:
            qi_scores = qi_scores / qi_max

        hi_max = hi_scores.max()
        if hi_max > 0:
            hi_scores = hi_scores / hi_max

        return qi_scores, hi_scores, qi_raw_max, hi_raw_max

    def _compute_per_head_markov_importance(
        self,
        attn_weights_per_head: torch.Tensor,
        query_idx: int,
        sink_size: int = 4,
        num_iterations: int = 10,
        gamma: float = 1.0,
        head_chunk_size: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        v7.0.0: Per-Head Markov Importance - Principled Per-Head Token Selection.

        Computes QI and HI scores INDEPENDENTLY for each attention head using
        Neumann series on each head's transition matrix. This preserves head
        specialization: different heads attend to different tokens and should
        keep different tokens in their KV cache.

        Key Insight:
        - SnapKV's per-head selection works because heads specialize
        - Our global QI/HI destroys this by forcing uniform token selection
        - Per-head Markov importance respects head specialization while
          maintaining the principled Markov chain framework

        Memory Efficiency:
        - Full computation would need O(num_heads × seq_len²) memory
        - Chunked processing: O(chunk_size × seq_len²) per chunk
        - At seq_len=32k, chunk_size=8: 8 × 32k² × 4 = 32GB (fits H100)

        Args:
            attn_weights_per_head: [bsz, num_heads, window_size, seq_len] attention
            query_idx: Query position (typically seq_len - 1)
            sink_size: Absorbing boundary size (first N tokens)
            num_iterations: Neumann series iterations
            gamma: Spectral decay factor (1.0 = no decay)
            head_chunk_size: Number of heads to process in parallel (memory vs speed)

        Returns:
            Tuple of (qi_per_head, hi_per_head, qi_raw_max, hi_raw_max)
            - qi_per_head, hi_per_head: each [num_heads, seq_len] (normalized)
            - qi_raw_max, hi_raw_max: float max values BEFORE normalization
        """
        # Average over batch dimension, keep heads separate
        # attn_weights_per_head: [bsz, num_heads, window_size, seq_len]
        attn = attn_weights_per_head.mean(dim=0)  # [num_heads, window_size, seq_len]
        num_heads, window_size, seq_len = attn.shape
        device = attn.device

        # Build full attention matrix from window attention
        # We need [num_heads, seq_len, seq_len] but only have window
        # Reconstruct causal attention: each position attends to previous positions
        # For positions beyond window, attention is approximated from last window row

        # Handle edge cases
        if seq_len <= sink_size:
            uniform = torch.ones(num_heads, seq_len, device=device) / seq_len
            return uniform, uniform.clone(), 1.0, 1.0

        n_transient = seq_len - sink_size
        query_transient_idx = query_idx - sink_size
        if query_transient_idx < 0:
            uniform = torch.ones(num_heads, seq_len, device=device) / seq_len
            return uniform, uniform.clone(), 1.0, 1.0

        # Initialize output tensors
        qi_per_head = torch.zeros(num_heads, seq_len, device=device, dtype=torch.float32)
        hi_per_head = torch.zeros(num_heads, seq_len, device=device, dtype=torch.float32)

        # Track raw max values across all chunks (before normalization)
        global_qi_raw_max = 0.0
        global_hi_raw_max = 0.0

        # Process heads in chunks for memory efficiency
        for chunk_start in range(0, num_heads, head_chunk_size):
            chunk_end = min(chunk_start + head_chunk_size, num_heads)
            chunk_size = chunk_end - chunk_start

            # Build full attention matrix for this chunk of heads
            # full_attn_chunk: [chunk_size, seq_len, seq_len]
            # We place window attention in the LAST window_size rows (where we have data)
            full_attn_chunk = torch.zeros(chunk_size, seq_len, seq_len, device=device, dtype=torch.float32)

            # Get window attention for this chunk: [chunk_size, window_size, seq_len]
            window_attn = attn[chunk_start:chunk_end, :, :]

            # Place window attention in the last window_size rows
            # These are the rows for positions [seq_len - window_size, ..., seq_len - 1]
            actual_window = min(window_size, seq_len)
            full_attn_chunk[:, -actual_window:, :] = window_attn[:, -actual_window:, :]

            # Make it properly causal: zero out future positions
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            full_attn_chunk.masked_fill_(causal_mask.unsqueeze(0), 0)

            # Normalize rows to get transition matrix P
            # Rows with all zeros (outside window) will get uniform distribution
            row_sums = full_attn_chunk.sum(dim=-1, keepdim=True)
            # For rows outside window, use uniform over previous positions
            zero_rows = (row_sums.squeeze(-1) < 1e-8)  # [chunk, seq_len]
            for i in range(seq_len - actual_window):
                if i > 0:  # Position 0 has no previous positions
                    full_attn_chunk[:, i, :i] = 1.0 / i
            row_sums = full_attn_chunk.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            P_chunk = full_attn_chunk / row_sums

            # Extract Q (transient-to-transient transitions)
            Q_chunk = P_chunk[:, sink_size:, sink_size:].contiguous()  # [chunk, n_transient, n_transient]

            # Initialize QI vectors: one-hot at query position
            v_qi = torch.zeros(chunk_size, n_transient, device=device, dtype=torch.float32)
            v_qi[:, query_transient_idx] = 1.0
            result_qi = v_qi.clone()

            # Initialize HI vectors: uniform distribution
            v_hi = torch.ones(chunk_size, n_transient, device=device, dtype=torch.float32) / n_transient
            result_hi = v_hi.clone()

            # Neumann series iteration: N = I + γQ + γ²Q² + ... + γ^k Q^k
            # v = Q^T @ v iteratively computes (Q^T)^k @ v₀
            for _ in range(num_iterations):
                # Batched matrix-vector multiply: [chunk, n, n] @ [chunk, n, 1] -> [chunk, n, 1]
                v_qi = gamma * torch.bmm(Q_chunk.transpose(-1, -2), v_qi.unsqueeze(-1)).squeeze(-1)
                v_hi = gamma * torch.bmm(Q_chunk.transpose(-1, -2), v_hi.unsqueeze(-1)).squeeze(-1)
                result_qi = result_qi + v_qi
                result_hi = result_hi + v_hi

            # Map back to full sequence [chunk, seq_len]
            qi_full = torch.zeros(chunk_size, seq_len, device=device, dtype=torch.float32)
            qi_full[:, sink_size:] = result_qi
            qi_full[:, :sink_size] = result_qi.sum(dim=-1, keepdim=True) * 0.01  # Small score for sink

            hi_full = torch.zeros(chunk_size, seq_len, device=device, dtype=torch.float32)
            hi_full[:, sink_size:] = result_hi
            hi_full[:, :sink_size] = result_hi.sum(dim=-1, keepdim=True) * 0.01

            # Capture raw max BEFORE normalization (for Layer Fairness debug)
            chunk_qi_raw_max = qi_full.max().item()
            chunk_hi_raw_max = hi_full.max().item()
            global_qi_raw_max = max(global_qi_raw_max, chunk_qi_raw_max)
            global_hi_raw_max = max(global_hi_raw_max, chunk_hi_raw_max)

            # Normalize each head's scores to [0, 1]
            qi_max = qi_full.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            qi_full = qi_full / qi_max

            hi_max = hi_full.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            hi_full = hi_full / hi_max

            # Store results for this chunk
            qi_per_head[chunk_start:chunk_end] = qi_full
            hi_per_head[chunk_start:chunk_end] = hi_full

        return qi_per_head, hi_per_head, global_qi_raw_max, global_hi_raw_max

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

    def _get_union_keep_mask(
        self,
        qi_scores: torch.Tensor,
        hi_scores: torch.Tensor,
        budget: int,
        seq_len: int,
        qi_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        v5.0.0: Union Selection - guarantee coverage from both QI and HI.

        Split budget between QI and HI, take union of top tokens from each.
        This ensures we keep tokens important by EITHER metric without
        dilution from multiplication.

        Key insight: QI and HI capture DIFFERENT tokens:
        - QI → late-position tokens (81% late) on paths FROM query
        - HI → early-position tokens (67% early) that are global hubs

        By taking union, we guarantee coverage from both perspectives.

        Args:
            qi_scores: Query Importance scores [seq_len]
            hi_scores: Hub Importance scores [seq_len]
            budget: Total tokens to keep
            seq_len: Current sequence length
            qi_ratio: Fraction of budget for QI (default 0.5)

        Returns:
            Boolean mask [seq_len] where True = keep
        """
        device = qi_scores.device

        # 1. Handle Budget (always static in LongBench)
        target_count = int(budget)

        # 2. Safety Constraints: Always keep Sink + Local
        min_required = self.sink_size + self.window_size
        if target_count < min_required:
            target_count = min_required

        # 3. Initialize mask with forced keep (Sink + Local)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask[:self.sink_size] = True  # Keep Sink

        local_start = max(self.sink_size, seq_len - self.window_size)
        mask[local_start:] = True  # Keep Local

        current_kept = mask.sum().item()
        remaining_budget = max(0, target_count - current_kept)

        if remaining_budget <= 0:
            return mask

        # 4. Compute QI and HI budgets
        qi_budget = int(remaining_budget * qi_ratio)
        hi_budget = remaining_budget - qi_budget

        # 5. Get candidates (exclude already kept positions)
        candidate_mask = ~mask
        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]

        if len(candidate_indices) == 0:
            return mask

        # Get scores for candidates only
        qi_candidate = qi_scores[candidate_indices]
        hi_candidate = hi_scores[candidate_indices]

        # 6. Select top-k by QI
        qi_k = min(qi_budget, len(candidate_indices))
        _, qi_top_local = torch.topk(qi_candidate, qi_k)
        qi_selected = set(candidate_indices[qi_top_local].cpu().tolist())

        # 7. Select top-k by HI
        hi_k = min(hi_budget, len(candidate_indices))
        _, hi_top_local = torch.topk(hi_candidate, hi_k)
        hi_selected = set(candidate_indices[hi_top_local].cpu().tolist())

        # 8. Take union
        union_selected = qi_selected | hi_selected

        # 9. Fill remaining slots if overlap exists (use HI for filling)
        if len(union_selected) < remaining_budget:
            # Get more candidates from HI (next best after already selected)
            shortfall = remaining_budget - len(union_selected)
            hi_all_k = min(len(candidate_indices), hi_budget + shortfall + qi_budget)
            _, hi_extended_local = torch.topk(hi_candidate, hi_all_k)
            hi_extended = candidate_indices[hi_extended_local].cpu().tolist()

            for idx in hi_extended:
                if idx not in union_selected:
                    union_selected.add(idx)
                    if len(union_selected) >= remaining_budget:
                        break

        # 10. Apply selection to mask
        for idx in union_selected:
            mask[idx] = True

        return mask

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

        # Layer Fairness Debug: Initialize raw score trackers
        _raw_max_qi = 0.0
        _raw_max_hi = 0.0

        mode = "Neumann" if self.use_neumann else "RandomWalk"
        if self.combination_mode == "union":
            comb = f"Union(QI:{self.qi_ratio:.0%},HI:{1-self.qi_ratio:.0%})"  # v5.0: Union selection (no DA)
        elif self.combination_mode == "union_da":
            comb = f"Union_DA(QI:{self.qi_ratio:.0%},HI:{1-self.qi_ratio:.0%})"  # v5.1: Union + DA weighting
        elif self.combination_mode == "dis":
            comb = "MAX(HI,QI)"  # v4.3: Both from fundamental matrix N
        elif self.combination_mode == "weighted":
            comb = f"weighted({self.h2o_weight:.1f})"
        else:
            comb = "MAX(H2O,QI)"  # v4.0: H2O from A, QI from N
        evict_mode = "PerHead" if self.per_head_eviction else "Shared"
        ablation_info = ""
        if self.ablate_qi:
            ablation_info = " [HI-only]"
        elif self.ablate_hi:
            ablation_info = " [QI-only]"
        if self.combination_mode == "union_da":
            version = "v5.1.0"
            da_note = ", DA-weighted"
        elif self.combination_mode == "union":
            version = "v5.0.0"
            da_note = ""
        elif self.combination_mode == "dis":
            # v6.0.0: No DA, v6.1.0: With smoothing, v6.2.0: Asymmetric Gaussian
            qi_k = self.qi_kernel_size if self.qi_kernel_size != 0 else self.smoothing_kernel
            hi_k = self.hi_kernel_size if self.hi_kernel_size != 0 else self.smoothing_kernel
            if self.smooth_hi_only:
                qi_k = 0  # Disabled

            if self.use_gaussian or self.smooth_hi_only or (qi_k != hi_k and (qi_k > 1 or hi_k > 1)):
                # v6.2.0: Asymmetric Gaussian, v6.3.0: Deep Field, v6.4.0: Transition Sharpening
                kernel_type = "gauss" if self.use_gaussian else "box"
                if self.neumann_temperature != 1.0:
                    # v6.4.0: Transition Sharpening - prevents signal diffusion
                    version = "v6.4.0"
                    da_note = f", {kernel_type}(QI={qi_k if qi_k > 0 else 'raw'}, HI={hi_k}), temp={self.neumann_temperature}"
                elif self.sink_size > 4 or self.neumann_iterations > 10:
                    # v6.3.0: Deep Field - extended absorbing boundary
                    version = "v6.3.0"
                    da_note = f", {kernel_type}(QI={qi_k if qi_k > 0 else 'raw'}, HI={hi_k}), sink={self.sink_size}"
                else:
                    version = "v6.2.0"
                    da_note = f", {kernel_type}(QI={qi_k if qi_k > 0 else 'raw'}, HI={hi_k})"
            elif self.smoothing_kernel > 1:
                version = "v6.1.0"
                da_note = f", smooth={self.smoothing_kernel}"
            else:
                version = "v6.0.0"
                da_note = ""  # v6.0.0: NO DA weighting (pure Markov)
        else:
            version = "v4.5.0"
            da_note = ", DA-weighted"
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

        # v4.4.0: Save per-head attention for per-head eviction
        # attn_weights: [bsz, num_heads, window_size, seq_len]
        attn_weights_per_head = attn_weights  # Keep reference before aggregation

        # =====================================================================
        # v6.5.0: Entropy-Aware Head Selection
        # v6.5.1: Transient Mass Selection (alternative to entropy)
        # Sharp heads (low entropy) OR high transient mass heads are used for QI
        # =====================================================================
        full_attn_hi = None  # Will be set if entropy_aware=True

        if self.entropy_aware:
            num_heads = attn_weights.shape[1]
            k = min(self.top_k_heads, num_heads)

            if self.head_selection_mode == "mass":
                # v6.5.1: Transient Mass Selection
                # Select heads with highest content-to-content connectivity
                head_mass = self._compute_head_transient_mass(attn_weights, self.sink_size)
                _, selected_head_indices = torch.topk(head_mass, k, largest=True)  # Highest mass
            else:
                # v6.5.0: Entropy Selection (default)
                # Select heads with lowest entropy (sharpest attention)
                head_entropy = self._compute_head_entropy(attn_weights)
                _, selected_head_indices = torch.topk(head_entropy, k, largest=False)  # Lowest entropy

            # attn_avg_qi: Max-pool over selected heads only (for QI)
            # MAX preserves strong "needle" signals from sharp heads
            attn_selected_heads = attn_weights[:, selected_head_indices, :, :]  # [bsz, k, window, seq]
            attn_avg_qi = attn_selected_heads.max(dim=1).values.mean(dim=0)  # [window, seq]

            # v6.7.0: HI pooling mode selection
            # v6.8.0: Mass-filtered hubs - filter dead heads before MEAN pooling
            # v6.8.1: Top-K mass heads - select top-k heads by mass for HI
            # v6.9.0: CDF-based head selection - select heads covering X% of total mass
            if self.hi_pooling_mode == "mean":
                # MEAN captures consensus/hub structure, prevents sharp heads from dominating
                # This fixes TREC/passage_count by preserving "broad context" heads
                if self.hi_mass_cdf > 0:
                    # v6.9.0: Select heads until cumulative mass >= threshold
                    # Adaptive: uses fewer heads if mass is concentrated, more if distributed
                    head_mass = self._compute_head_transient_mass(attn_weights, self.sink_size)
                    total_mass = head_mass.sum()
                    if total_mass > 0:
                        # Sort heads by mass descending
                        sorted_mass, sorted_indices = head_mass.sort(descending=True)
                        # Compute cumulative sum normalized (CDF)
                        cumsum = sorted_mass.cumsum(dim=0) / total_mass
                        # Find how many heads needed to reach threshold
                        n_heads_needed = (cumsum < self.hi_mass_cdf).sum().item() + 1
                        n_heads_needed = min(n_heads_needed, head_mass.shape[0])
                        # Select top heads by mass
                        selected_indices = sorted_indices[:n_heads_needed]
                        attn_selected = attn_weights[:, selected_indices, :, :]
                        attn_avg_hi = attn_selected.mean(dim=1).mean(dim=0)  # [window, seq]
                    else:
                        # Fallback: all heads have zero mass
                        attn_avg_hi = attn_weights.mean(dim=1).mean(dim=0)
                elif self.hi_top_k_heads > 0:
                    # v6.8.1: Select top-k heads by transient mass for HI
                    # More principled than threshold - always uses exactly k high-connectivity heads
                    head_mass = self._compute_head_transient_mass(attn_weights, self.sink_size)
                    num_heads = head_mass.shape[0]
                    k = min(self.hi_top_k_heads, num_heads)
                    _, top_k_indices = head_mass.topk(k)  # [k]
                    attn_top_k = attn_weights[:, top_k_indices, :, :]  # [bsz, k, window, seq]
                    attn_avg_hi = attn_top_k.mean(dim=1).mean(dim=0)  # [window, seq]
                elif self.hi_mass_threshold > 0:
                    # v6.8.0: Filter out dead heads (heads with low transient mass)
                    # Dead heads mostly attend to sink tokens and add noise to consensus
                    head_mass = self._compute_head_transient_mass(attn_weights, self.sink_size)
                    alive_mask = head_mass >= self.hi_mass_threshold  # [num_heads]
                    n_alive = alive_mask.sum().item()
                    num_heads = head_mass.shape[0]

                    # Diagnostic logging for head stats
                    if getattr(self, 'hi_log_head_stats', False):
                        mass_sorted, _ = head_mass.sort(descending=True)
                        print(f"[HI-DIAG] layer={getattr(self, '_layer_idx', '?')} "
                              f"n_heads={num_heads} n_alive={n_alive} "
                              f"threshold={self.hi_mass_threshold:.2f} "
                              f"mass_range=[{head_mass.min().item():.3f}, {head_mass.max().item():.3f}] "
                              f"mass_mean={head_mass.mean().item():.3f} "
                              f"top5_mass={mass_sorted[:5].tolist()}")

                    if n_alive > 0:
                        # MEAN over alive heads only
                        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
                        attn_alive = attn_weights[:, alive_indices, :, :]  # [bsz, n_alive, window, seq]
                        attn_avg_hi = attn_alive.mean(dim=1).mean(dim=0)  # [window, seq]
                    else:
                        # Fallback: all heads are dead, use all anyway
                        attn_avg_hi = attn_weights.mean(dim=1).mean(dim=0)
                else:
                    # v6.7 behavior: MEAN over all heads
                    attn_avg_hi = attn_weights.mean(dim=1).mean(dim=0)  # [window, seq]
            else:
                # v6.5 behavior: MAX pooling (sharp heads dominate)
                attn_avg_hi = attn_weights.max(dim=1).values.mean(dim=0)  # [window, seq]

            # Use selected-head attention for the main flow (QI matrix)
            attn_avg = attn_avg_qi
        else:
            # Default: use all heads for both QI and HI
            attn_avg = attn_weights.max(dim=1).values.mean(dim=0)  # Max over heads, avg over batch
            attn_avg_hi = None

        # Build full attention matrix approximation (used for QI)
        # For positions outside the window, use uniform causal attention
        full_attn = torch.zeros(q_len, q_len, device=key_states.device, dtype=torch.float32)

        # Fill the window portion with actual attention
        full_attn[-self.window_size:, :] = attn_avg

        # For positions outside window, use H2O-WEIGHTED transitions (not uniform)
        # This guides walkers toward high-attention tokens instead of random backward jumps
        n_prefix = q_len - self.window_size

        def _build_prefix_transitions(h2o_source, device):
            """Helper to build H2O-weighted prefix transitions."""
            h2o_scores = h2o_source.sum(dim=0)  # [seq_len]
            h2o_prefix = h2o_scores[:n_prefix].clone().clamp(min=1e-6)
            cumsum = h2o_prefix.cumsum(dim=0)
            denom = torch.zeros(n_prefix, device=device, dtype=torch.float32)
            denom[1:] = cumsum[:-1]
            denom[0] = 1.0
            h2o_expanded = h2o_prefix.unsqueeze(0).expand(n_prefix, n_prefix)
            denom_expanded = denom.unsqueeze(1).expand(n_prefix, n_prefix)
            h2o_trans = h2o_expanded / (denom_expanded + 1e-8)
            mask = torch.tril(torch.ones(n_prefix, n_prefix, device=device, dtype=torch.float32), diagonal=-1)
            return h2o_trans * mask

        if n_prefix > 1:
            full_attn[:n_prefix, :n_prefix] = _build_prefix_transitions(attn_avg, key_states.device)

        # v6.5.0: Build separate full_attn_hi for HI if entropy_aware
        if self.entropy_aware and attn_avg_hi is not None:
            full_attn_hi = torch.zeros(q_len, q_len, device=key_states.device, dtype=torch.float32)
            full_attn_hi[-self.window_size:, :] = attn_avg_hi
            if n_prefix > 1:
                full_attn_hi[:n_prefix, :n_prefix] = _build_prefix_transitions(attn_avg_hi, key_states.device)

        # =====================================================================
        # STEP 2: Compute Importance Scores
        # v4.2.0: Dual-Importance Scoring (DIS) - QI × HI from fundamental matrix
        # v4.0.0: Deterministic Neumann series (default)
        # v3.0.0: Stochastic random walks (legacy, for ablation)
        # =====================================================================
        current_idx = q_len - 1

        if self.combination_mode == "union":
            # v5.0.0: Union Selection - guarantee coverage from both QI and HI
            # Key insight: QI and HI capture DIFFERENT tokens (0.589 correlation)
            # - QI → late-position tokens (81% late) on paths FROM query
            # - HI → early-position tokens (67% early) that are global hubs
            # By taking union, we guarantee coverage from both perspectives.
            qi_scores, hi_scores, _raw_max_qi, _raw_max_hi = self._compute_dual_importance_scores(
                full_attn[:q_len, :q_len].contiguous(),
                current_idx,
                sink_size=self.sink_size,
                num_iterations=self.neumann_iterations,
                temperature=self.neumann_temperature,
                attention_for_hi=full_attn_hi[:q_len, :q_len].contiguous() if full_attn_hi is not None else None,
                gamma=self.neumann_gamma,
            )

            # Store QI and HI for union selection (called in STEP 3)
            self._qi_scores_for_union = qi_scores
            self._hi_scores_for_union = hi_scores

            # For compatibility with scoring path, use MAX(rank(QI), rank(HI))
            # This is only used for debug logging, not for selection
            qi_rank = self._rank_normalize(qi_scores)

            # v6.12.0/v6.12.1: HI Signal Gating/Scaling - handle noisy HI in deep layers
            hi_scale_by_max = getattr(self, 'hi_scale_by_max', False)
            hi_signal_threshold = getattr(self, 'hi_signal_threshold', 0.0)
            if hi_scale_by_max:
                # v6.12.1: Soft scaling - scale rank by layer's raw max
                # Automatically dampens noisy layers without threshold tuning
                hi_rank = self._rank_normalize(hi_scores) * _raw_max_hi
                if getattr(self, 'hi_log_head_stats', False):
                    print(f"[HI-SCALE] layer={getattr(self, '_layer_idx', '?')} "
                          f"scaled by max={_raw_max_hi:.4f}")
            elif hi_signal_threshold > 0 and _raw_max_hi < hi_signal_threshold:
                # v6.12.0: Hard gating - silence layer entirely
                hi_rank = torch.zeros_like(hi_scores)
                if getattr(self, 'hi_log_head_stats', False):
                    print(f"[HI-GATE] layer={getattr(self, '_layer_idx', '?')} "
                          f"silenced (max={_raw_max_hi:.4f} < {hi_signal_threshold})")
            else:
                hi_rank = self._rank_normalize(hi_scores)

            combined_scores = torch.maximum(qi_rank, hi_rank)

            # Store for debugging
            influence_scores_full = torch.zeros(
                full_attn.shape[0], device=full_attn.device, dtype=torch.float32
            )
            influence_scores_full[:q_len] = qi_scores
            influence_scores = influence_scores_full

        elif self.combination_mode == "union_da":
            # v5.1.0: Union Selection WITH DA weighting
            # Same as union but multiply by Direct Attention to preserve query relevance.
            # This keeps the attention grounding that helped v4.5.0 on TREC/classification.
            qi_scores, hi_scores, _raw_max_qi, _raw_max_hi = self._compute_dual_importance_scores(
                full_attn[:q_len, :q_len].contiguous(),
                current_idx,
                sink_size=self.sink_size,
                num_iterations=self.neumann_iterations,
                temperature=self.neumann_temperature,
                attention_for_hi=full_attn_hi[:q_len, :q_len].contiguous() if full_attn_hi is not None else None,
                gamma=self.neumann_gamma,
            )

            # Compute Direct Attention (DA) - what the window actually attends to
            da_scores = full_attn[-self.window_size:, :q_len].sum(dim=0)
            da_scores = da_scores.clamp(min=1e-8)

            # Weight QI and HI by DA before union selection
            qi_weighted = qi_scores * da_scores
            hi_weighted = hi_scores * da_scores

            # Store DA-weighted scores for union selection
            self._qi_scores_for_union = qi_weighted
            self._hi_scores_for_union = hi_weighted

            # For compatibility with scoring path
            qi_rank = self._rank_normalize(qi_weighted)

            # v6.12.0/v6.12.1: HI Signal Gating/Scaling - handle noisy HI in deep layers
            hi_scale_by_max = getattr(self, 'hi_scale_by_max', False)
            hi_signal_threshold = getattr(self, 'hi_signal_threshold', 0.0)
            if hi_scale_by_max:
                # v6.12.1: Soft scaling - scale rank by layer's raw max
                hi_rank = self._rank_normalize(hi_weighted) * _raw_max_hi
                if getattr(self, 'hi_log_head_stats', False):
                    print(f"[HI-SCALE] layer={getattr(self, '_layer_idx', '?')} "
                          f"scaled by max={_raw_max_hi:.4f}")
            elif hi_signal_threshold > 0 and _raw_max_hi < hi_signal_threshold:
                # v6.12.0: Hard gating - silence layer entirely
                hi_rank = torch.zeros_like(hi_weighted)
                if getattr(self, 'hi_log_head_stats', False):
                    print(f"[HI-GATE] layer={getattr(self, '_layer_idx', '?')} "
                          f"silenced (max={_raw_max_hi:.4f} < {hi_signal_threshold})")
            else:
                hi_rank = self._rank_normalize(hi_weighted)

            combined_scores = torch.maximum(qi_rank, hi_rank)

            # Store for debugging
            influence_scores_full = torch.zeros(
                full_attn.shape[0], device=full_attn.device, dtype=torch.float32
            )
            influence_scores_full[:q_len] = qi_scores
            influence_scores = influence_scores_full

        elif self.combination_mode == "dis":
            # v6.0.0: Pure Dual-Importance Scoring (NO DA weighting)
            # QI and HI from fundamental matrix N, combined via MAX(rank)
            # Analysis shows DA weighting HURTS performance: 42.24 (with DA) vs 42.42 (without DA)
            # DA hurts multi-hop QA (multifieldqa -1.06, narrativeqa -0.89) and retrieval tasks
            qi_scores, hi_scores, _raw_max_qi, _raw_max_hi = self._compute_dual_importance_scores(
                full_attn[:q_len, :q_len].contiguous(),
                current_idx,
                sink_size=self.sink_size,
                num_iterations=self.neumann_iterations,
                temperature=self.neumann_temperature,
                attention_for_hi=full_attn_hi[:q_len, :q_len].contiguous() if full_attn_hi is not None else None,
                gamma=self.neumann_gamma,
            )
            # _raw_max_qi and _raw_max_hi are now returned from the function (pre-normalization)

            # v6.2.0: Asymmetric Gaussian Smoothing for Frequency Separation
            #
            # The "Frequency Separation" Hypothesis:
            # - QI (Query Importance) = "High Frequency" signal: finds precise needles (keywords, code tokens)
            # - HI (Hub Importance) = "Low Frequency" signal: captures broad context/topics
            #
            # Strategy: "Sharp QI, Smooth HI"
            # - Sharp QI: Preserves exact matches for Code/Classification (RepoBench, TREC)
            # - Smooth HI: Captures phrase context for Summarization/Narrative (GovReport, NarrativeQA)
            #
            # Gaussian vs Boxcar:
            # - Boxcar [0.2, 0.2, 0.2, 0.2, 0.2]: Flattens spike (80% penalty to center)
            # - Gaussian [0.1, 0.2, 0.4, 0.2, 0.1]: Center stays dominant, neighbors get lift

            # Determine effective kernel sizes (asymmetric smoothing)
            qi_k = self.qi_kernel_size if self.qi_kernel_size != 0 else self.smoothing_kernel
            hi_k = self.hi_kernel_size if self.hi_kernel_size != 0 else self.smoothing_kernel

            # Handle smooth_hi_only flag (legacy compatibility)
            if self.smooth_hi_only and qi_k > 1:
                qi_k = 0  # Disable QI smoothing

            # Apply asymmetric smoothing
            if qi_k > 1:
                # QI: Tight smoothing (or raw) for precision
                qi_sigma = self.gaussian_sigma * 0.5 if self.use_gaussian else 1.0  # Tighter sigma for QI
                qi_scores = self._apply_smoothing(qi_scores, qi_k, use_gaussian=self.use_gaussian, sigma=qi_sigma)

            if hi_k > 1:
                # HI: Wider smoothing for context preservation
                hi_scores = self._apply_smoothing(hi_scores, hi_k, use_gaussian=self.use_gaussian, sigma=self.gaussian_sigma)

            # v6.10.0: Bridge Importance via A² (second-order attention)
            # Captures 2-hop paths for cross-document reasoning (2WikiMQA, multi-hop QA)
            bi_scores = None
            if getattr(self, 'use_bridge_importance', False):
                # Use HI attention matrix (broader consensus) for bridge computation
                attn_for_bi = full_attn_hi[:q_len, :q_len] if full_attn_hi is not None else full_attn[:q_len, :q_len]
                bi_scores = self._compute_bridge_importance(attn_for_bi)

                # Apply smoothing to BI (bridges need context)
                bi_k = getattr(self, 'bi_kernel_size', 5)
                if bi_k > 1:
                    bi_scores = self._apply_smoothing(bi_scores, bi_k, use_gaussian=self.use_gaussian, sigma=self.gaussian_sigma)

            # v6.0.0: Direct rank normalization WITHOUT DA weighting
            # Pure Markov chain signals preserve transitive reasoning paths

            qi_rank = self._rank_normalize(qi_scores)

            # v6.12.0/v6.12.1: HI Signal Gating/Scaling - handle noisy HI in deep layers
            # If raw HI max is below threshold, the HI scores are just noise that would
            # be stretched to [0,1] by rank normalization, evicting valid QI signals.
            hi_scale_by_max = getattr(self, 'hi_scale_by_max', False)
            hi_signal_threshold = getattr(self, 'hi_signal_threshold', 0.0)
            if hi_scale_by_max:
                # v6.12.1: Soft scaling - scale rank by layer's raw max
                # Automatically dampens noisy layers without threshold tuning
                hi_rank = self._rank_normalize(hi_scores) * _raw_max_hi
                if getattr(self, 'hi_log_head_stats', False):
                    print(f"[HI-SCALE] layer={getattr(self, '_layer_idx', '?')} "
                          f"scaled by max={_raw_max_hi:.4f}")
            elif hi_signal_threshold > 0 and _raw_max_hi < hi_signal_threshold:
                # v6.12.0: Hard gating - silence layer entirely
                hi_rank = torch.zeros_like(hi_scores)
                if getattr(self, 'hi_log_head_stats', False):
                    print(f"[HI-GATE] layer={getattr(self, '_layer_idx', '?')} "
                          f"silenced (max={_raw_max_hi:.4f} < {hi_signal_threshold})")
            else:
                hi_rank = self._rank_normalize(hi_scores)

            # v6.10.0: Rank normalize BI if enabled
            bi_rank = None
            if bi_scores is not None:
                bi_rank = self._rank_normalize(bi_scores)

            # v4.3.1: Ablation support - zero out one signal for A1 experiment
            if self.ablate_qi:
                # HI-only ablation: use only Hub Importance
                qi_rank = torch.zeros_like(qi_rank)
            if self.ablate_hi:
                # QI-only ablation: use only Query Importance
                hi_rank = torch.zeros_like(hi_rank)

            # v6.10.0: MAX of pure Markov signals (with optional BI for cross-document reasoning)
            combined_scores = torch.maximum(qi_rank, hi_rank)
            if bi_rank is not None:
                combined_scores = torch.maximum(combined_scores, bi_rank)

            # Store for debugging (use influence_scores variable for compatibility)
            influence_scores_full = torch.zeros(
                full_attn.shape[0], device=full_attn.device, dtype=torch.float32
            )
            influence_scores_full[:q_len] = qi_scores  # Store QI as "influence" for debug
            influence_scores = influence_scores_full

        elif self.use_neumann:
            # v4.0.0: Deterministic influence via Neumann series
            # Computes expected visit counts analytically: N = (I-Q)^(-1)
            influence_scores = self._compute_influence_neumann(
                full_attn[:q_len, :q_len].contiguous(),
                current_idx,
                sink_size=self.sink_size,
                num_iterations=self.neumann_iterations,
                temperature=self.neumann_temperature,
                gamma=self.neumann_gamma,
            )
            # Pad to full buffer size
            influence_scores_full = torch.zeros(
                full_attn.shape[0], device=full_attn.device, dtype=torch.float32
            )
            influence_scores_full[:q_len] = influence_scores
            influence_scores = influence_scores_full

            # =====================================================================
            # STEP 2b: MAX(H2O, Influence) with Rank Normalization
            # - H2O = column sums (good for NarrativeQA-style simple retrieval)
            # - Influence = walker scores (good for HotpotQA-style multi-hop)
            # - MAX = hedging strategy that works for both
            # =====================================================================
            # Compute H2O scores (column sums)
            h2o_scores = full_attn.sum(dim=0)  # [seq_len]

            # Rank normalize both to [0, 1] (puts them on equal footing)
            h2o_rank = self._rank_normalize(h2o_scores[:q_len])
            influence_rank = self._rank_normalize(influence_scores[:q_len])

            # v4.1.0: Configurable combination mode
            if self.combination_mode == "weighted":
                # Weighted average: α * H2O + (1-α) * Influence
                combined_scores = self.h2o_weight * h2o_rank + (1 - self.h2o_weight) * influence_rank
            else:
                # MAX combination: keeps token if EITHER force wants it (default)
                combined_scores = torch.maximum(h2o_rank, influence_rank)

        else:
            # v3.0.0 (legacy): Stochastic random walks via CUDA kernel
            self._graph.update_and_step_influence_walker(
                full_attn.contiguous(),
                current_idx,
                self.num_walkers,    # 10000 walkers
                self.max_steps,      # 10 steps
                self.sink_size,      # Absorb at first 4 tokens
            )
            influence_scores = self._graph.get_influence_scores()

            # Compute H2O scores (column sums)
            h2o_scores = full_attn.sum(dim=0)  # [seq_len]

            # Rank normalize both to [0, 1] (puts them on equal footing)
            h2o_rank = self._rank_normalize(h2o_scores[:q_len])
            influence_rank = self._rank_normalize(influence_scores[:q_len])

            # MAX combination
            combined_scores = torch.maximum(h2o_rank, influence_rank)

        # v3.0.0 Breakthrough 2: Instruction Anchor Detection
        # Detect and boost instruction-anchoring tokens for few-shot tasks
        # NOTE: Heuristic method uses attention patterns, doesn't need tokenizer
        if self.use_instruction_anchors:
            try:
                # Detect anchors using attention pattern heuristics
                self._instruction_anchors = self._detect_instruction_anchors_heuristic(
                    full_attn, q_len
                )
                # Boost instruction anchors to ensure they're kept
                num_boosted = 0
                for anchor_pos in self._instruction_anchors:
                    if 0 <= anchor_pos < q_len:
                        combined_scores[anchor_pos] = 1.0
                        num_boosted += 1

                # Debug: Log anchor detection results
                if num_boosted > 0:
                    print(f"  [v3.0.0] Detected {len(self._instruction_anchors)} instruction anchors, boosted {num_boosted}")
            except Exception as e:
                print(f"  [v3.0.0] Anchor detection failed: {e}")

        # Expand to full buffer size
        scores = torch.zeros_like(influence_scores)
        scores[:q_len] = combined_scores

        # =====================================================================
        # STEP 3: EVICTION BASED ON SCORES
        # v4.4.0: Support for per-head eviction (like SnapKV)
        # =====================================================================

        if self.per_head_eviction:
            # =====================================================================
            # v7.0.0: PER-HEAD MARKOV IMPORTANCE (FIXED)
            # Each head computes its own QI/HI using Neumann series on its own
            # attention pattern, then selects tokens via MAX(QI, HI).
            #
            # Key fix: Build full attention matrix per head using ALL window rows,
            # not just the last row. For positions outside window, use uniform
            # attention over previous positions.
            # =====================================================================

            non_window_len = q_len - self.window_size

            # Compute per-head Markov importance scores
            # qi_per_head, hi_per_head: [num_heads, seq_len]
            # _raw_max_qi, _raw_max_hi: raw max values BEFORE normalization
            qi_per_head, hi_per_head, _raw_max_qi, _raw_max_hi = self._compute_per_head_markov_importance(
                attn_weights_per_head,
                query_idx=q_len - 1,
                sink_size=self.sink_size,
                num_iterations=self.neumann_iterations,
                gamma=self.neumann_gamma,
                head_chunk_size=getattr(self, 'head_chunk_size', 4),
            )

            # Truncate to non-window portion
            qi_per_head = qi_per_head[:, :non_window_len]  # [num_heads, non_window_len]
            hi_per_head = hi_per_head[:, :non_window_len]  # [num_heads, non_window_len]

            # Rank normalize per head
            qi_rank_per_head = torch.zeros_like(qi_per_head)
            hi_rank_per_head = torch.zeros_like(hi_per_head)
            for h in range(num_heads):
                qi_rank_per_head[h] = self._rank_normalize(qi_per_head[h])
                hi_rank_per_head[h] = self._rank_normalize(hi_per_head[h])

            # Per-head MAX(QI, HI)
            if self.ablate_hi and not self.ablate_qi:
                per_head_scores = qi_rank_per_head
            elif not self.ablate_hi and self.ablate_qi:
                per_head_scores = hi_rank_per_head
            elif self.ablate_hi and self.ablate_qi:
                # Both ablated: pure per-head H2O (SnapKV-style)
                h2o_per_head = attn_weights_per_head[:, :, :, :non_window_len].sum(dim=2)
                h2o_per_head = h2o_per_head.mean(dim=0)  # [num_heads, non_window_len]
                per_head_scores = torch.zeros_like(h2o_per_head)
                for h in range(num_heads):
                    per_head_scores[h] = self._rank_normalize(h2o_per_head[h])
            else:
                # Default: MAX(QI, HI) per head
                per_head_scores = torch.maximum(qi_rank_per_head, hi_rank_per_head)

            # Apply Gaussian smoothing for spatial coherence (optional)
            smooth_kernel = max(self.qi_kernel_size, self.hi_kernel_size)
            if smooth_kernel > 1 and self.use_gaussian:
                per_head_scores_smooth = torch.zeros_like(per_head_scores)
                for h in range(num_heads):
                    per_head_scores_smooth[h] = self._apply_smoothing(
                        per_head_scores[h],
                        smooth_kernel,
                        use_gaussian=True,
                        sigma=self.gaussian_sigma
                    )
                per_head_scores = per_head_scores_smooth

            # Per-head top-k selection
            # Budget for non-window tokens
            non_window_budget = self.max_capacity_prompt - self.window_size - self.sink_size

            # Always keep sink tokens (first sink_size)
            # For the middle portion (between sink and window), do per-head selection
            middle_len = non_window_len - self.sink_size
            if middle_len > 0 and non_window_budget > 0:
                # Get scores for middle portion (excluding sink)
                middle_scores = per_head_scores[:, self.sink_size:]  # [num_heads, middle_len]

                # Top-k per head
                num_select = min(non_window_budget, middle_len)
                _, top_indices_per_head = middle_scores.topk(num_select, dim=-1)  # [num_heads, num_select]

                # Adjust indices to account for sink offset
                top_indices_per_head = top_indices_per_head + self.sink_size  # [num_heads, num_select]

                # Build full index tensor: sink + selected middle tokens
                # For each head: [0, 1, ..., sink_size-1, selected_indices...]
                sink_indices = torch.arange(self.sink_size, device=key_states.device)
                sink_indices = sink_indices.unsqueeze(0).expand(num_heads, -1)  # [num_heads, sink_size]

                # Concatenate sink + per-head selected indices
                indices_per_head = torch.cat([sink_indices, top_indices_per_head], dim=-1)  # [num_heads, sink_size + num_select]
            else:
                # Edge case: just keep sink
                indices_per_head = torch.arange(self.sink_size, device=key_states.device)
                indices_per_head = indices_per_head.unsqueeze(0).expand(num_heads, -1)

            # Sort indices for each head (for gather operation)
            indices_per_head, _ = indices_per_head.sort(dim=-1)

            # Expand for gather: [bsz, num_heads, num_keep, head_dim]
            num_keep_per_head = indices_per_head.shape[-1]
            indices_expanded = indices_per_head.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, -1, head_dim)

            # Gather per-head (different tokens per head)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices_expanded)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices_expanded)

            # Append local window
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]

            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)

            # Layer Fairness Diagnostic for per-head mode
            if self.debug:
                layer_idx = _circuitkv_debug_next_layer()
                # For per-head, count average middle tokens kept across heads
                middle_kept = num_select if middle_len > 0 and non_window_budget > 0 else 0
                _record_layer_fairness(layer_idx, middle_kept, _raw_max_qi, _raw_max_hi, 0.0)

        else:
            # =====================================================================
            # Original shared eviction (same tokens for all heads)
            # =====================================================================
            # Compute keep mask using static budgeting
            if self.combination_mode in ["union", "union_da"]:
                # v5.0.0/v5.1.0: Union selection - use stored QI and HI scores
                # For union_da, these are already DA-weighted
                keep_mask = self._get_union_keep_mask(
                    self._qi_scores_for_union,
                    self._hi_scores_for_union,
                    self.max_capacity_prompt,
                    q_len,
                    qi_ratio=self.qi_ratio,
                )
            else:
                keep_mask = self._get_keep_mask(
                    scores,
                    self.max_capacity_prompt,
                    q_len
                )

            # Layer Fairness Diagnostic: record per-layer stats and detailed debug
            if self.debug:
                layer_idx = _circuitkv_debug_next_layer()
                # Count only MIDDLE tokens (between sink and window) for Layer Fairness test
                kept_positions = keep_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                middle_kept = sum(1 for p in kept_positions if self.sink_size <= p < q_len - self.window_size)
                # Use raw max scores captured before smoothing/normalization
                raw_max_combined = scores.max().item() if scores is not None else 0.0
                _record_layer_fairness(layer_idx, middle_kept, _raw_max_qi, _raw_max_hi, raw_max_combined)

                # Detailed debug logging (only for layer 1)
                h2o_scores_debug = full_attn.sum(dim=0)
                qi_debug = locals().get('qi_scores', None)
                hi_debug = locals().get('hi_scores', None)
                self._log_debug(
                    q_len, h2o_scores_debug, scores, keep_mask, full_attn,
                    qi_debug, hi_debug, attn_weights_per_head, layer_idx
                )

            # Apply eviction - shared across all heads
            # Get indices of tokens to keep (excluding local window which is appended)
            non_local_mask = keep_mask.clone()
            non_local_mask[-self.window_size:] = False
            keep_indices = non_local_mask.nonzero(as_tuple=True)[0]

            # Number of non-local tokens to keep
            num_keep = keep_indices.shape[0]

            # Gather selected tokens (same indices for all heads)
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
        # RC+B: Bidirectional Circuit Walks (disabled - hurt narrativeqa score)
        if not hasattr(self.config, 'bidirectional'):
            self.config.bidirectional = False
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
        # v4.0.0: Deterministic Neumann Series
        if not hasattr(self.config, 'use_neumann'):
            self.config.use_neumann = True  # Default: use deterministic Neumann
        if not hasattr(self.config, 'neumann_iterations'):
            self.config.neumann_iterations = 10  # Neumann series iterations
        if not hasattr(self.config, 'neumann_temperature'):
            self.config.neumann_temperature = 1.0  # Temperature for attention sharpening
        # v4.1.0: Combination tuning
        if not hasattr(self.config, 'h2o_weight'):
            self.config.h2o_weight = 0.5  # Weight for H2O in weighted combination
        if not hasattr(self.config, 'combination_mode'):
            self.config.combination_mode = "dis"  # Default: DIS v6.0.0 (no DA, pure Markov)
        # v4.2.0: Dual-Importance Scoring
        if not hasattr(self.config, 'dis_alpha'):
            self.config.dis_alpha = 0.5  # QI weight in DIS (0.5 = symmetric)
        # v4.3.1: Ablation flags for A1 experiment
        if not hasattr(self.config, 'ablate_qi'):
            self.config.ablate_qi = False  # If True, disable QI (use HI only)
        if not hasattr(self.config, 'ablate_hi'):
            self.config.ablate_hi = False  # If True, disable HI (use QI only)
        # v7.0.0: Per-head Markov importance
        if not hasattr(self.config, 'per_head_eviction'):
            self.config.per_head_eviction = False  # If True, each head computes its own QI/HI
        if not hasattr(self.config, 'head_chunk_size'):
            self.config.head_chunk_size = 8  # Number of heads to process in parallel
        # v5.0.0: Union Selection
        if not hasattr(self.config, 'qi_ratio'):
            self.config.qi_ratio = 0.5  # Ratio of budget for QI in Union mode
        # v6.1.0: Smoothing Kernel
        if not hasattr(self.config, 'smoothing_kernel'):
            self.config.smoothing_kernel = 0  # 0=disabled, 5=recommended for phrase preservation
        # v6.2.0: Asymmetric Gaussian Smoothing
        if not hasattr(self.config, 'smooth_hi_only'):
            self.config.smooth_hi_only = False  # If True, only smooth HI (keep QI sharp)
        if not hasattr(self.config, 'use_gaussian'):
            self.config.use_gaussian = False  # If True, use Gaussian kernel instead of boxcar
        if not hasattr(self.config, 'qi_kernel_size'):
            self.config.qi_kernel_size = 0  # 0=use smoothing_kernel, -1=raw/no smoothing
        if not hasattr(self.config, 'hi_kernel_size'):
            self.config.hi_kernel_size = 0  # 0=use smoothing_kernel
        if not hasattr(self.config, 'gaussian_sigma'):
            self.config.gaussian_sigma = 1.0  # Sigma for Gaussian kernel

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
        # Debug
        debug=self.config.circuitkv_debug,
        # v4.0.0: Deterministic Neumann Series
        use_neumann=self.config.use_neumann,
        neumann_iterations=self.config.neumann_iterations,
        neumann_temperature=self.config.neumann_temperature,
        neumann_gamma=getattr(self.config, 'neumann_gamma', 1.0),
        # v4.1.0: Combination tuning
        h2o_weight=self.config.h2o_weight,
        combination_mode=self.config.combination_mode,
        # v4.2.0: Dual-Importance Scoring
        dis_alpha=self.config.dis_alpha,
        # v4.3.1: Ablation flags
        ablate_qi=self.config.ablate_qi,
        ablate_hi=self.config.ablate_hi,
        # v7.0.0: Per-head Markov importance
        per_head_eviction=self.config.per_head_eviction,
        head_chunk_size=self.config.head_chunk_size,
        # v5.0.0: Union Selection
        qi_ratio=self.config.qi_ratio,
        # v6.1.0: Smoothing Kernel
        smoothing_kernel=self.config.smoothing_kernel,
        # v6.2.0: Asymmetric Gaussian Smoothing
        smooth_hi_only=self.config.smooth_hi_only,
        use_gaussian=self.config.use_gaussian,
        qi_kernel_size=self.config.qi_kernel_size,
        hi_kernel_size=self.config.hi_kernel_size,
        gaussian_sigma=self.config.gaussian_sigma,
        # v6.5.0: Entropy-Aware Head Selection
        entropy_aware=getattr(self.config, 'entropy_aware', False),
        top_k_heads=getattr(self.config, 'top_k_heads', 8),
        # v6.5.1: Head Selection Mode
        head_selection_mode=getattr(self.config, 'head_selection_mode', 'entropy'),
        # v6.7.0: HI Pooling Mode
        hi_pooling_mode=getattr(self.config, 'hi_pooling_mode', 'mean'),
        # v6.8.0: Mass-Filtered Hubs
        hi_mass_threshold=getattr(self.config, 'hi_mass_threshold', 0.0),
        # v6.8.1: Top-K Mass Heads for HI
        hi_top_k_heads=getattr(self.config, 'hi_top_k_heads', 0),
        # v6.9.0: CDF-Based Head Selection
        hi_mass_cdf=getattr(self.config, 'hi_mass_cdf', 0.0),
        # Diagnostic logging
        hi_log_head_stats=getattr(self.config, 'hi_log_head_stats', False),
        # v6.10.0: Bridge Importance via A²
        use_bridge_importance=getattr(self.config, 'use_bridge_importance', False),
        bi_kernel_size=getattr(self.config, 'bi_kernel_size', 5),
    )