import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Callable, Tuple, Dict, Any


class PerceiverAttention(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            num_heads=8,
            bias=True,
            concat_query_to_kv=False,
            init_weights="truncnormal002",
            init_last_proj_zero=False,
    ):
        super().__init__()
        assert hasattr(F, "scaled_dot_product_attention")
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.concat_query_to_kv = concat_query_to_kv
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero

        self.kv = nn.Linear(kv_dim or dim, dim * 2, bias=bias)
        self.q = nn.Linear(dim, dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
            init_xavier_uniform_merged_linear(self.kv, num_layers=2)
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError
        if self.init_last_proj_zero:
            nn.init.zeros_(self.proj.weight)
            # init_weights == "torch" has no zero bias init
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, q, kv, attn_mask=None):
        # project to attention space
        if self.concat_query_to_kv:
            kv = torch.concat([kv, q], dim=1)
        kv = self.kv(kv)
        q = self.q(q)

        # split per head
        q = einops.rearrange(
            q,
            "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k, v = einops.rearrange(
            kv,
            "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        return x


class PerceiverBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            kv_dim=None,
            mlp_hidden_dim=None,
            drop_path=0.,
            act_ctor=nn.GELU,
            norm_ctor=nn.LayerNorm,
            bias=True,
            concat_query_to_kv=False,
            layerscale=None,
            eps=1e-6,
            init_weights="xavier_uniform",
            init_norms="nonaffine",
            init_last_proj_zero=False,
    ):
        super().__init__()
        self.init_norms = init_norms
        mlp_hidden_dim = mlp_hidden_dim or dim * 4
        self.norm1q = norm_ctor(dim, eps=eps)
        self.norm1kv = norm_ctor(kv_dim or dim, eps=eps)
        self.attn = PerceiverAttention1d(
            dim=dim,
            num_heads=num_heads,
            kv_dim=kv_dim,
            bias=bias,
            concat_query_to_kv=concat_query_to_kv,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.ls1 = nn.Identity() if layerscale is None else LayerScale(dim, init_scale=layerscale)
        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp = Mlp(
            in_dim=dim,
            hidden_dim=mlp_hidden_dim,
            bias=bias,
            act_ctor=act_ctor,
            init_weights=init_weights,
            init_last_proj_zero=init_last_proj_zero,
        )
        self.ls2 = nn.Identity() if layerscale is None else LayerScale(dim, init_scale=layerscale)
        self.drop_path2 = DropPath(drop_prob=drop_path)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_norms == "torch":
            pass
        elif self.init_norms == "nonaffine":
            init_norms_as_noaffine(self.norm1q)
            init_norms_as_noaffine(self.norm1kv)
            init_norms_as_noaffine(self.norm2)
        else:
            raise NotImplementedError

    def _attn_residual_path(self, q, kv, attn_mask):
        return self.ls1(self.attn(q=self.norm1q(q), kv=self.norm1kv(kv), attn_mask=attn_mask))

    def _mlp_residual_path(self, x):
        return self.ls2(self.mlp(self.norm2(x)))

    def forward(self, q, kv, attn_mask=None):
        q = self.drop_path1(
            q,
            residual_path=self._attn_residual_path,
            residual_path_kwargs=dict(kv=kv, attn_mask=attn_mask),
        )
        q = self.drop_path2(q, self._mlp_residual_path)
        return q


class DecoderPerceiver(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            ndim,
            dim,
            depth,
            num_heads,
            unbatch_mode="dense_to_sparse_unpadded",
            perc_dim=None,
            perc_num_heads=None,
            cond_dim=None,
            init_weights="truncnormal002",
            **kwargs,
    ):
        super().__init__(**kwargs)
        perc_dim = perc_dim or dim
        perc_num_heads = perc_num_heads or num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights
        self.unbatch_mode = unbatch_mode

        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        # blocks
        if cond_dim is None:
            block_ctor = VitBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    init_weights=init_weights,
                )
                for _ in range(depth)
            ],
        )

        # prepare perceiver
        self.pos_embed = ContinuousSincosEmbed(
            dim=perc_dim,
            ndim=ndim,
        )
        if cond_dim is None:
            block_ctor = PerceiverBlock
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=cond_dim)

        # decoder
        self.query_proj = nn.Sequential(
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
        )
        self.perc = block_ctor(dim=perc_dim, kv_dim=dim, num_heads=perc_num_heads, init_weights=init_weights)
        self.pred = nn.Sequential(
            nn.LayerNorm(perc_dim, eps=1e-6),
            LinearProjection(perc_dim, output_dim, init_weights=init_weights),
        )

    def forward(self, x, output_pos, condition=None):
        # check inputs
        assert x.ndim == 3, "expected shape (batch_size, num_latent_tokens, dim)"
        assert output_pos.ndim == 3, "expected shape (batch_size, num_outputs, dim) num_outputs might be padded"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        # create query
        query = self.pos_embed(output_pos)
        query = self.query_proj(query)

        x = self.perc(q=query, kv=x, **cond_kwargs)
        x = self.pred(x)
        if self.unbatch_mode == "dense_to_sparse_unpadded":
            # dense to sparse where no padding needs to be considered
            x = einops.rearrange(
                x,
                "batch_size seqlen dim -> (batch_size seqlen) dim",
            )
        elif self.unbatch_mode == "image":
            # rearrange to square image
            height = math.sqrt(x.size(1))
            assert height.is_integer()
            x = einops.rearrange(
                x,
                "batch_size (height width) dim -> batch_size dim height width",
                height=int(height),
            )
        else:
            raise NotImplementedError(f"invalid unbatch_mode '{self.unbatch_mode}'")

        return x


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
        assert pos.ndim == 3  # (batch_size, num_points, ndim)
        assert pos.shape[2] == self.ndim
        
        batch_size, num_points = pos.shape[0], pos.shape[1]
        
        # Create sinusoidal position embeddings
        # Each spatial dimension gets dim/ndim features
        emb_dim_per_dim = self.dim // self.ndim
        half_emb_dim = emb_dim_per_dim // 2
        
        emb = jnp.log(10000.0) / (half_emb_dim - 1)
        emb = jnp.exp(jnp.arange(half_emb_dim) * -emb)
        
        # Apply to each spatial dimension separately
        embeddings = []
        for i in range(self.ndim):
            # Get position for this dimension
            pos_i = pos[:, :, i:i+1]  # Keep dim for broadcasting (batch_size, num_points, 1)
            
            # Create embeddings for this dimension
            emb_i = pos_i * emb[None, None, None, :]  # (batch_size, num_points, 1, half_emb_dim)
            emb_i = jnp.concatenate([jnp.sin(emb_i), jnp.cos(emb_i)], axis=-1)  # (batch_size, num_points, 1, emb_dim_per_dim)
            emb_i = emb_i.reshape(batch_size, num_points, 2 * half_emb_dim)  # (batch_size, num_points, emb_dim_per_dim)
            
            embeddings.append(emb_i)
        
        # Concatenate embeddings from all dimensions
        x = jnp.concatenate(embeddings, axis=-1)  # (batch_size, num_points, emb_dim_per_dim * ndim)
        
        # If we have leftover dimensions due to integer division, pad with zeros
        if x.shape[-1] < self.dim:
            padding = self.dim - x.shape[-1]
            x = jnp.pad(x, ((0, 0), (0, 0), (0, padding)))
            
        return x


class PrenormBlock(nn.Module):
    dim: int
    num_heads: int
    
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


def batch_k_nearest_neighbors(x, supernode_idxs, k):
    """JAX implementation of k-nearest neighbors with batch support
    
    Args:
        x: Shape (batch_size, num_points, ndim)
        supernode_idxs: Shape (batch_size, num_supernodes) - Indices of supernodes
        k: Number of nearest neighbors to consider
        
    Returns:
        coords: Shape (batch_size, num_supernodes, k, 3) - Coordinates of k nearest neighbors for each supernode
    """    
    def process_sample(sample_x):
        # Compute pairwise distances
        dists = jnp.sqrt(jnp.sum((sample_x[:, None, :] - sample_x[None, :, :]) ** 2, axis=-1))
        dists_from_supernodes = dists[supernode_idxs]
        
        # Get indices of k nearest neighbors for each supernode
        neighbor_idxs = jnp.argsort(dists_from_supernodes, axis=1)[:, :k]

        return sample_x[neighbor_idxs]
    
    # Process each sample in the batch
    return jax.vmap(process_sample)(x)


def batch_segment_aggregation(src, idx, num_segments, reduce="mean"):
    """JAX implementation of segment aggregation with batch support
    
    Args:
        src: Shape (batch_size, num_elements, feature_dim)
        idx: Shape (batch_size, num_elements) - indices for aggregation
        num_segments: Number of segments to aggregate into (must be an int, not an array)
        reduce: Aggregation method ('mean', 'sum', 'max')
        
    Returns:
        aggregated: Shape (batch_size, num_segments, feature_dim)
    """
    batch_size, num_elements, feature_dim = src.shape
    
    def aggregate_sample(sample_src, sample_idx, sample_num_segments):
        # Create mask for valid indices (not -1)
        valid_mask = sample_idx >= 0
        
        # Clip indices to valid range for one-hot encoding
        # -1 indices will be mapped to 0 but then masked out
        clipped_idx = jnp.maximum(sample_idx, 0)
        
        # Create one-hot encoding of segment indices
        one_hot = jax.nn.one_hot(clipped_idx, sample_num_segments)  # (num_elements, num_segments)
        
        # Apply mask to one-hot encoding
        one_hot = one_hot * valid_mask[:, None]
        
        # Reshape for broadcasting
        one_hot = one_hot[:, :, None]  # (num_elements, num_segments, 1)
        sample_src = sample_src[:, None, :]  # (num_elements, 1, feature_dim)
        
        # Compute aggregate by segment
        if reduce == "mean":
            # Mask and sum elements
            masked = one_hot * sample_src  # (num_elements, num_segments, feature_dim)
            summed = jnp.sum(masked, axis=0)  # (num_segments, feature_dim)
            
            # Compute counts for normalization
            counts = jnp.sum(one_hot, axis=0)  # (num_segments, 1)
            counts = jnp.maximum(counts, 1.0)  # Avoid division by zero
            
            # Normalize
            result = summed / counts
            
        elif reduce == "sum":
            masked = one_hot * sample_src
            result = jnp.sum(masked, axis=0)
            
        elif reduce == "max":
            # Set values to -inf where segment doesn't apply
            masked = jnp.where(one_hot > 0, sample_src, -jnp.inf)
            result = jnp.max(masked, axis=0)
            
            # Handle case where segment has no elements
            valid_segments = jnp.sum(one_hot, axis=0) > 0
            result = jnp.where(valid_segments, result, 0.0)
            
        else:
            raise ValueError(f"Unknown reduce operation: {reduce}")
            
        return result
    
    # Apply aggregation to each sample in batch, passing num_segments to each sample
    # Use vmap with in_axes to specify which arguments have batch dimension
    return jax.vmap(aggregate_sample, in_axes=(0, 0, None))(src, idx, num_segments)


class SupernodePooling(nn.Module):
    max_degree: int
    input_dim: int
    hidden_dim: int
    init_weights: str = "torch"
    
    @nn.compact
    def __call__(self, input_pos, supernode_idxs):
        """
        Args:
            input_pos: Shape (batch_size, num_points, ndim)
            supernode_idxs: Shape (batch_size, num_supernodes) - Indices of supernodes
            
        Returns:
            x: Shape (batch_size, max_supernodes, hidden_dim)
        """
        
        # Radius graph - creates edges between nodes
        supernode_neighbors_points = batch_k_nearest_neighbors(
            x=input_pos,
            supernode_idxs=supernode_idxs,
            k=self.max_degree,
        )
        
        # Embed mesh
        input_proj = LinearProjection(features=self.hidden_dim, init_weights=self.init_weights)(supernode_neighbors_points)
        pos_embed = ContinuousSincosEmbed(dim=self.hidden_dim, ndim=self.ndim)(supernode_neighbors_points)
        
        x = input_proj + pos_embed
        
        # Message passing network
        message_net = nn.Sequential([
            LinearProjection(features=self.hidden_dim, init_weights=self.init_weights),
            lambda x: nn.gelu(x),
            LinearProjection(features=self.hidden_dim, init_weights=self.init_weights),
        ])

        x = message_net(x)
        x = jnp.mean(x, axis=-2)
        
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
    max_supernodes: int  # New parameter for fixed supernode count
    perc_dim: Optional[int] = None
    perc_num_heads: Optional[int] = None
    num_latent_tokens: Optional[int] = None
    cond_dim: Optional[int] = None
    init_weights: str = "truncnormal"
    output_coord_dim: Optional[int] = None
    coord_enc_dim: Optional[int] = None
    coord_enc_depth: Optional[int] = 2
    coord_enc_num_heads: Optional[int] = 4
    
    def setup(self):
        # Supernode pooling with fixed max_supernodes
        self.supernode_pooling = SupernodePooling(
            max_degree=self.max_degree,
            input_dim=self.input_dim,
            hidden_dim=self.gnn_dim
        )
        
        # Encoder projection
        self.enc_proj = LinearProjection(
            features=self.enc_dim, 
            init_weights=self.init_weights,
            optional=True
        )
        
        # Transformer blocks for conditioning
        self.enc_blocks = [
            PrenormBlock(
                dim=self.enc_dim, 
                num_heads=self.enc_num_heads, 
                init_weights=self.init_weights
            )
            for _ in range(self.enc_depth)
        ]
        
        # Perceiver pooling for conditioning
        if self.num_latent_tokens is not None:
            self.perceiver = PerceiverPoolingBlock(
                dim=self.perc_dim,
                num_heads=self.perc_num_heads,
                num_query_tokens=self.num_latent_tokens,
                perceiver_kwargs=dict(
                    kv_dim=self.enc_dim,
                    init_weights=self.init_weights,
                ),
            )
        else:
            self.perceiver = nn.Identity()


        self.encoder_token = self.param(
            'encoder_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.enc_dim)
        )
        
        # Transformer blocks for coordinate transformation
        self.coord_enc_blocks = [
            PrenormBlock(
                dim=self.coord_enc_dim, 
                num_heads=self.coord_enc_num_heads, 
                init_weights=self.init_weights
            )
            for _ in range(self.coord_enc_depth)
        ]
        
        # Final projection to output coordinate dimensions
        self.coord_output_proj = nn.Dense(
            features=self.output_coord_dim,
            kernel_init=nn.initializers.truncated_normal(stddev=0.02) # n.initializers.glorot_uniform() #
        )
    
    def __call__(self, points, supernode_idxs):
        """
        Args in two possible modes:
        Mode 1 (separate feature and position):
            input_feat_or_points: Shape (batch_size, num_points, input_dim) - Feature tensor
            input_pos: Shape (batch_size, num_points, ndim) - Position tensor 
            supernode_mask: Shape (batch_size, num_points) - Boolean mask for supernodes
            condition: Optional Shape (batch_size, cond_dim)
            
        Mode 2 (combined points, for compatibility with UniversalAutoencoder):
            input_feat_or_points: Shape (batch_size, num_points, ndim) - Position tensor
            input_pos: Used as supernode_mask, shape (batch_size, num_points)
            supernode_mask: Used as condition, optional shape (batch_size, cond_dim)
            condition: Ignored
            
        Returns:
            Tuple containing:
            - x: Conditioning tensor, shape depends on perceiver setting:
               - With perceiver: (batch_size, num_latent_tokens, perc_dim)
               - Without perceiver: (batch_size, num_supernodes, enc_dim)
            - coords: Transformed coordinates, shape (batch_size, num_points, output_coord_dim)
        """

        # ----- Conditioning branch -----
        # Supernode pooling
        x = self.supernode_pooling(
            input_points=points,
            supernode_idxs=supernode_idxs,
        )
        
        # Project to encoder dimension
        x = self.enc_proj(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply perceiver if needed
        if self.perceiver is not None:
            x = self.perceiver(kv=x)
        
        # ----- Coordinate transformation branch -----
        if self.output_coord_dim is not None:
            # Combine feature and position information
            batch_size, num_points = input_feat.shape[0], input_feat.shape[1]
            
            # Project input features and add position embedding
            coord_x = self.coord_input_proj(input_feat) + self.coord_pos_embed(input_pos)
            
            # Apply transformer blocks
            for block in self.coord_blocks:
                if condition is not None:
                    coord_x = block(coord_x, cond=condition)
                else:
                    coord_x = block(coord_x)
            
            # Project to output coordinate dimensions
            output_coords = self.coord_output_proj(coord_x)
            
            # Ensure coordinates are in a reasonable range [-1, 1]
            output_coords = jnp.tanh(output_coords)
        else:
            # If no coordinate transformation is specified, use original coordinates
            output_coords = input_pos
        
        return x, output_coords


def test_batch_encoder_supernodes():
    """Test function for BatchEncoderSupernodes"""

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define test parameters
    batch_size = 4
    num_points = 1000  # Reduced from 1000 to make shapes smaller for debugging
    num_supernodes = 10
    input_dim = 3
    ndim = 3
    output_coord_dim = 2  # Transform from 3D to 2D
    max_degree = 5  # Reduce this to make shapes more manageable
    
    print(f"Parameters: batch_size={batch_size}, num_points={num_points}, num_supernodes={num_supernodes}")
    print(f"input_dim={input_dim}, ndim={ndim}, max_degree={max_degree}, output_coord_dim={output_coord_dim}")
    
    # Create random input data
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    
    # Create input features and positions with batch dimension
    input_feat = jax.random.normal(subkey1, (batch_size, num_points, input_dim))
    input_pos = jax.random.normal(subkey2, (batch_size, num_points, ndim))
    
    # Create supernode mask (randomly select some points as supernodes)
    all_indices = jnp.arange(num_points)
    supernode_indices = jax.random.choice(
        subkey3, 
        all_indices, 
        shape=(num_supernodes,), 
        replace=False
    )
    supernode_mask = jnp.zeros((batch_size, num_points), dtype=bool)
    supernode_mask = supernode_mask.at[:, supernode_indices].set(True)
    
    print(f"Input shapes: input_feat={input_feat.shape}, input_pos={input_pos.shape}")
    print(f"supernode_mask={supernode_mask.shape}, num_true={jnp.sum(supernode_mask)}")
    
    # Optional condition
    key, subkey4 = jax.random.split(key)
    condition = jax.random.normal(subkey4, (batch_size, 16))
    
    # Initialize model with coordinate transformation and fixed max_supernodes
    model = EncoderSupernodes(
        input_dim=input_dim,
        ndim=ndim,
        radius=0.5,
        max_degree=max_degree,
        gnn_dim=32,
        enc_dim=64,
        enc_depth=2,
        enc_num_heads=4,
        max_supernodes=32,  # Fixed maximum number of supernodes
        perc_dim=64,
        perc_num_heads=4,
        num_latent_tokens=8,
        # cond_dim=16,
        init_weights="truncnormal",
        output_coord_dim=output_coord_dim,
        coord_enc_dim=48,
        coord_enc_depth=2,
        coord_enc_num_heads=4
    )
    
    try:
        # Initialize parameters
        key, subkey = jax.random.split(key)
        params = model.init(subkey, input_feat, input_pos, supernode_mask) # , condition)
        
        # Apply model (Mode 1 - separate feature and position)
        output, output_coords = model.apply(params, input_feat, input_pos, supernode_mask) # , condition)
        
        print("\nModel initialized and forward pass completed successfully! (Mode 1)")
        print(f"Conditioning output shape: {output.shape}")
        print(f"Expected conditioning shape: (batch_size, num_latent_tokens, perc_dim) = ({batch_size}, 8, 64)")
        print(f"Coordinate output shape: {output_coords.shape}")
        print(f"Expected coordinate shape: (batch_size, num_points, output_coord_dim) = ({batch_size}, {num_points}, {output_coord_dim})")
        
        # Test without coordinate transformation
        model_no_coord_transform = EncoderSupernodes(
            input_dim=input_dim,
            ndim=ndim,
            radius=0.5,
            max_degree=max_degree,
            gnn_dim=32,
            enc_dim=64,
            enc_depth=2,
            enc_num_heads=4,
            perc_dim=64,
            perc_num_heads=4,
            num_latent_tokens=8,
            # cond_dim=16,
            init_weights="truncnormal",
            # No output_coord_dim specified, so should just pass through the input coordinates
            max_supernodes=32
        )
        
        key, subkey = jax.random.split(key)
        params_no_coord = model_no_coord_transform.init(subkey, input_feat, input_pos, supernode_mask) # , condition)
        output_no_coord, coords_no_transform = model_no_coord_transform.apply(params_no_coord, input_feat, input_pos, supernode_mask) # , condition)
        
        print("\nModel without coordinate transformation tested successfully!")
        print(f"Conditioning output shape: {output_no_coord.shape}")
        print(f"Expected conditioning shape: (batch_size, num_latent_tokens, perc_dim) = ({batch_size}, 8, 64)")
        print(f"Coordinate output shape: {coords_no_transform.shape}")
        print(f"Expected coordinate shape: (batch_size, num_points, ndim) = ({batch_size}, {num_points}, {ndim})")
        
        # Verify that the output coordinates are in the expected range [-1, 1]
        coord_min = jnp.min(output_coords)
        coord_max = jnp.max(output_coords)
        print(f"\nOutput coordinate range: [{coord_min}, {coord_max}]")
        print(f"Expected range: [-1, 1] (enforced by tanh activation)")
        
        # Test Mode 2 (points-only for compatibility with UniversalAutoencoder)
        points = input_pos  # Use positions as points
        
        # Initialize parameters for Mode 2
        key, subkey = jax.random.split(key)
        params_mode2 = model.init(subkey, points)
        
        # Apply model in Mode 2
        output_mode2, output_coords_mode2 = model.apply(params_mode2, points)
        
        print("\nModel tested in Mode 2 (points-only) successfully!")
        print(f"Conditioning output shape: {output_mode2.shape}")
        print(f"Coordinate output shape: {output_coords_mode2.shape}")
        print(f"Expected coordinate shape: (batch_size, num_points, output_coord_dim) = ({batch_size}, {num_points}, {output_coord_dim})")
        
        return output, output_coords, output_mode2, output_coords_mode2
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    test_batch_encoder_supernodes() 