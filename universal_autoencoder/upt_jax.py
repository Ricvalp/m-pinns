import jax
import jax.numpy as jnp
import flax.linen as nn
import einops
from functools import partial
from typing import Optional, Callable, Tuple, Dict, Any

class LinearProjection(nn.Module):
    features: int
    use_bias: bool = True
    init_weights: str = "torch"
    optional: bool = False
    
    @nn.compact
    def __call__(self, x):
        if self.optional and x.shape[-1] == self.features:
            return x
            
        if self.init_weights == "torch":
            kernel_init = nn.initializers.lecun_normal()
        elif self.init_weights == "truncnormal":
            kernel_init = nn.initializers.truncated_normal(stddev=0.02)
        else:
            raise ValueError(f"Unknown init_weights: {self.init_weights}")
            
        return nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=kernel_init,
        )(x)

class ContinuousSincosEmbed(nn.Module):
    dim: int
    ndim: int
    
    @nn.compact
    def __call__(self, pos):
        assert pos.ndim == 2
        assert pos.shape[1] == self.ndim
        
        # Create sinusoidal position embeddings
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = pos[:, :, None] * emb[None, None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        emb = einops.rearrange(emb, 'b n d -> b (n d)')
        
        # If dim is odd, add an extra zero column
        if self.dim % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
            
        return emb

class PrenormBlock(nn.Module):
    dim: int
    num_heads: int
    init_weights: str = "torch"
    
    @nn.compact
    def __call__(self, x):
        # Self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(x, x)
        x = residual + x
        
        # MLP
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.dim)(x)
        x = residual + x
        
        return x

class DitBlock(nn.Module):
    dim: int
    num_heads: int
    cond_dim: int
    init_weights: str = "torch"
    
    @nn.compact
    def __call__(self, x, cond=None):
        # Self-attention with condition
        residual = x
        x = nn.LayerNorm()(x)
        
        # Condition projection for attention
        cond_attn = nn.Dense(features=self.dim)(cond)
        x = x + cond_attn[:, None, :]
        
        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(x, x)
        x = residual + x
        
        # MLP with condition
        residual = x
        x = nn.LayerNorm()(x)
        
        # Condition projection for MLP
        cond_mlp = nn.Dense(features=self.dim)(cond)
        x = x + cond_mlp[:, None, :]
        
        x = nn.Dense(features=self.dim * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.dim)(x)
        x = residual + x
        
        return x

class PerceiverPoolingBlock(nn.Module):
    dim: int
    num_heads: int
    num_query_tokens: int
    perceiver_kwargs: Dict[str, Any] = None
    
    @nn.compact
    def __call__(self, kv):
        # Initialize learnable query tokens
        query = self.param(
            'query',
            nn.initializers.normal(stddev=0.02),
            (1, self.num_query_tokens, self.dim)
        )
        
        # Expand query to batch size
        batch_size = kv.shape[0]
        query = jnp.tile(query, (batch_size, 1, 1))
        
        # Cross-attention
        kv_dim = self.perceiver_kwargs.get('kv_dim', self.dim)
        init_weights = self.perceiver_kwargs.get('init_weights', 'torch')
        
        # Project kv if dimensions don't match
        if kv_dim != self.dim:
            kv = LinearProjection(features=self.dim, init_weights=init_weights)(kv)
        
        # Apply cross-attention
        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(query, kv)
        
        return x

class DitPerceiverPoolingBlock(nn.Module):
    dim: int
    num_heads: int
    num_query_tokens: int
    perceiver_kwargs: Dict[str, Any] = None
    
    @nn.compact
    def __call__(self, kv, cond=None):
        # Initialize learnable query tokens
        query = self.param(
            'query',
            nn.initializers.normal(stddev=0.02),
            (1, self.num_query_tokens, self.dim)
        )
        
        # Expand query to batch size
        batch_size = kv.shape[0]
        query = jnp.tile(query, (batch_size, 1, 1))
        
        # Condition projection
        cond_dim = self.perceiver_kwargs.get('cond_dim')
        cond_proj = nn.Dense(features=self.dim)(cond)
        query = query + cond_proj[:, None, :]
        
        # Cross-attention
        kv_dim = self.perceiver_kwargs.get('kv_dim', self.dim)
        init_weights = self.perceiver_kwargs.get('init_weights', 'torch')
        
        # Project kv if dimensions don't match
        if kv_dim != self.dim:
            kv = LinearProjection(features=self.dim, init_weights=init_weights)(kv)
        
        # Apply cross-attention
        x = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02),
        )(query, kv)
        
        return x

def radius_graph_jax(x, r, batch, max_num_neighbors=32, loop=True):
    """JAX implementation of radius_graph"""
    batch_size = jnp.max(batch) + 1
    
    def process_batch(b_idx):
        # Get points for this batch
        mask = batch == b_idx
        batch_points = x[mask]
        batch_indices = jnp.where(mask)[0]
        
        # Compute pairwise distances
        dists = jnp.sqrt(jnp.sum((batch_points[:, None, :] - batch_points[None, :, :]) ** 2, axis=-1))
        
        # Create mask for points within radius
        radius_mask = dists <= r
        
        # Handle self-loops
        if not loop:
            radius_mask = radius_mask & (jnp.arange(len(batch_points))[:, None] != jnp.arange(len(batch_points))[None, :])
        
        # Get source and destination indices
        src_indices, dst_indices = jnp.where(radius_mask)
        
        # Map back to original indices
        src_indices = batch_indices[src_indices]
        dst_indices = batch_indices[dst_indices]
        
        # Limit number of neighbors if needed
        if max_num_neighbors is not None:
            # Sort by distance to prioritize closer neighbors
            sorted_idx = jnp.argsort(dists[radius_mask])
            sorted_idx = sorted_idx[:max_num_neighbors]
            src_indices = src_indices[sorted_idx]
            dst_indices = dst_indices[sorted_idx]
        
        return src_indices, dst_indices
    
    # Process each batch and concatenate results
    edges = jax.vmap(process_batch)(jnp.arange(batch_size))
    src_indices = jnp.concatenate(edges[0])
    dst_indices = jnp.concatenate(edges[1])
    
    return jnp.stack([src_indices, dst_indices])

def segment_csr(src, indptr, reduce="mean"):
    """JAX implementation of segment_csr"""
    # Extract segments based on indptr
    results = []
    for i in range(len(indptr) - 1):
        start, end = indptr[i], indptr[i+1]
        segment = src[start:end]
        
        if reduce == "mean":
            results.append(jnp.mean(segment, axis=0))
        elif reduce == "sum":
            results.append(jnp.sum(segment, axis=0))
        elif reduce == "max":
            results.append(jnp.max(segment, axis=0))
        else:
            raise ValueError(f"Unknown reduce operation: {reduce}")
    
    return jnp.stack(results)

class SupernodePooling(nn.Module):
    radius: float
    max_degree: int
    input_dim: int
    hidden_dim: int
    ndim: int
    init_weights: str = "torch"
    
    @nn.compact
    def __call__(self, input_feat, input_pos, supernode_idxs, batch_idx):
        assert input_feat.ndim == 2
        assert input_pos.ndim == 2
        assert supernode_idxs.ndim == 1
        
        # Radius graph
        input_edges = radius_graph_jax(
            x=input_pos,
            r=self.radius,
            max_num_neighbors=self.max_degree,
            batch=batch_idx,
            loop=True,
        )
        
        # Filter edges to only include supernode edges
        is_supernode_edge = jnp.isin(input_edges[0], supernode_idxs)
        input_edges = input_edges[:, is_supernode_edge]
        
        # Embed mesh
        input_proj = LinearProjection(features=self.hidden_dim, init_weights=self.init_weights)
        pos_embed = ContinuousSincosEmbed(dim=self.hidden_dim, ndim=self.ndim)
        x = input_proj(input_feat) + pos_embed(input_pos)
        
        # Create message input
        dst_idx, src_idx = input_edges[0], input_edges[1]
        message_input = jnp.concatenate([x[src_idx], x[dst_idx]], axis=1)
        
        # Message passing network
        message = nn.Sequential([
            LinearProjection(features=self.hidden_dim, init_weights=self.init_weights),
            lambda x: nn.gelu(x),
            LinearProjection(features=self.hidden_dim, init_weights=self.init_weights),
        ])
        x = message(message_input)
        
        # Accumulate messages
        dst_indices, counts = jnp.unique(dst_idx, return_counts=True)
        padded_counts = jnp.zeros(len(counts) + 1, dtype=counts.dtype)
        padded_counts = padded_counts.at[1:].set(counts)
        indptr = jnp.cumsum(padded_counts, axis=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")
        
        # Sanity check
        batch_size = jnp.max(batch_idx) + 1
        assert dst_indices.size % batch_size == 0
        
        # Convert to dense tensor
        num_supernodes = dst_indices.size // batch_size
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
            num_supernodes=num_supernodes,
        )
        
        return x

class EncoderSupernodes(nn.Module):
    input_dim: int
    ndim: int
    radius: float
    max_degree: int
    gnn_dim: int
    enc_dim: int
    enc_depth: int
    enc_num_heads: int
    perc_dim: Optional[int] = None
    perc_num_heads: Optional[int] = None
    num_latent_tokens: Optional[int] = None
    cond_dim: Optional[int] = None
    init_weights: str = "truncnormal"
    
    def setup(self):
        # Supernode pooling
        self.supernode_pooling = SupernodePooling(
            radius=self.radius,
            max_degree=self.max_degree,
            input_dim=self.input_dim,
            hidden_dim=self.gnn_dim,
            ndim=self.ndim,
        )
        
        # Encoder projection
        self.enc_proj = LinearProjection(
            features=self.enc_dim, 
            init_weights=self.init_weights, 
            optional=True
        )
        
        # Transformer blocks
        self.blocks = [
            DitBlock(
                dim=self.enc_dim, 
                num_heads=self.enc_num_heads, 
                cond_dim=self.cond_dim,
                init_weights=self.init_weights
            ) if self.cond_dim is not None else
            PrenormBlock(
                dim=self.enc_dim, 
                num_heads=self.enc_num_heads, 
                init_weights=self.init_weights
            )
            for _ in range(self.enc_depth)
        ]
        
        # # Perceiver pooling
        # if self.num_latent_tokens is not None:
        #     if self.cond_dim is None:
        #         self.perceiver = PerceiverPoolingBlock(
        #             dim=self.perc_dim,
        #             num_heads=self.perc_num_heads,
        #             num_query_tokens=self.num_latent_tokens,
        #             perceiver_kwargs=dict(
        #                 kv_dim=self.enc_dim,
        #                 init_weights=self.init_weights,
        #             ),
        #         )
        #     else:
        #         self.perceiver = DitPerceiverPoolingBlock(
        #             dim=self.perc_dim,
        #             num_heads=self.perc_num_heads,
        #             num_query_tokens=self.num_latent_tokens,
        #             perceiver_kwargs=dict(
        #                 kv_dim=self.enc_dim,
        #                 cond_dim=self.cond_dim,
        #                 init_weights=self.init_weights,
        #             ),
        #         )
        # else:
        
        self.perceiver = None
    

    def __call__(self, input_feat, input_pos, supernode_idxs, batch_idx, condition=None):
        # Check inputs
        assert input_feat.ndim == 2, "expected sparse tensor (batch_size * num_inputs, input_dim)"
        assert input_pos.ndim == 2, "expected sparse tensor (batch_size * num_inputs, ndim)"
        assert len(input_feat) == len(input_pos), "expected input_feat and input_pos to have same length"
        assert supernode_idxs.ndim == 1, "supernode_idxs is a 1D tensor of indices that are used as supernodes"
        assert batch_idx.ndim == 1, "batch_idx should be 1D tensor that assigns elements of the input to samples"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"
        
        # Supernode pooling
        x = self.supernode_pooling(
            input_feat=input_feat,
            input_pos=input_pos,
            supernode_idxs=supernode_idxs,
            batch_idx=batch_idx,
        )
        
        # Project to encoder dimension
        x = self.enc_proj(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            if condition is not None:
                x = block(x, cond=condition)
            else:
                x = block(x)
        
        # Apply perceiver if needed
        if self.perceiver is not None:
            if condition is not None:
                x = self.perceiver(kv=x, cond=condition)
            else:
                x = self.perceiver(kv=x)
        
        return x


def test_encoder_supernodes():
    """Test function for EncoderSupernodes"""

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define test parameters
    batch_size = 16
    num_points = 10000
    num_supernodes = 10
    input_dim = 3
    ndim = 3
    
    # Create random input data
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    
    # Create input features and positions
    input_feat = jax.random.normal(subkey1, (batch_size * num_points, input_dim))
    input_pos = jax.random.normal(subkey2, (batch_size * num_points, ndim))
    
    # Create batch indices
    batch_idx = jnp.repeat(jnp.arange(batch_size), num_points)
    
    # Create supernode indices (randomly select some points as supernodes)
    all_indices = jnp.arange(batch_size * num_points)
    supernode_idxs = jax.random.choice(
        subkey3, 
        all_indices, 
        shape=(batch_size * num_supernodes,), 
        replace=False
    )
    
    # Optional condition
    key, subkey4 = jax.random.split(key)
    condition = jax.random.normal(subkey4, (batch_size, 16))
    
    # Initialize model
    model = EncoderSupernodes(
        input_dim=input_dim,
        ndim=ndim,
        radius=0.5,
        max_degree=16,
        gnn_dim=32,
        enc_dim=64,
        enc_depth=2,
        enc_num_heads=4,
        perc_dim=64,
        perc_num_heads=4,
        num_latent_tokens=8,
        cond_dim=16,
        init_weights="truncnormal"
    )
    
    # Initialize parameters
    key, subkey = jax.random.split(key)
    params = model.init(subkey, input_feat, input_pos, supernode_idxs, batch_idx, condition)
    
    # Apply model
    output = model.apply(params, input_feat, input_pos, supernode_idxs, batch_idx, condition)
    
    print("Model initialized and forward pass completed successfully!")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (batch_size, num_latent_tokens, perc_dim) = ({batch_size}, 8, 64)")
    
    # Test without condition
    model_no_cond = EncoderSupernodes(
        input_dim=input_dim,
        ndim=ndim,
        radius=0.5,
        max_degree=16,
        gnn_dim=32,
        enc_dim=64,
        enc_depth=2,
        enc_num_heads=4,
        init_weights="truncnormal"
    )
    
    key, subkey = jax.random.split(key)
    params_no_cond = model_no_cond.init(subkey, input_feat, input_pos, supernode_idxs, batch_idx)
    output_no_cond = model_no_cond.apply(params_no_cond, input_feat, input_pos, supernode_idxs, batch_idx)
    
    print("\nModel without perceiver and condition tested successfully!")
    print(f"Output shape: {output_no_cond.shape}")
    print(f"Expected shape: (batch_size, num_supernodes_per_batch, enc_dim)")
    
    return output, output_no_cond

if __name__ == "__main__":
    test_encoder_supernodes() 