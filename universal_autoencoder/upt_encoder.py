import jax
import jax.numpy as jnp
import flax.linen as nn
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


def batch_radius_graph_jax(x, r, max_num_neighbors=32, loop=True):
    """JAX implementation of radius_graph with batch support
    
    Args:
        x: Shape (batch_size, num_points, ndim)
        r: Radius
        max_num_neighbors: Maximum number of neighbors per node
        loop: Whether to include self-loops
        
    Returns:
        edge_index: Shape (2, batch_size, num_edges)
    """
    batch_size, num_points, ndim = x.shape
    
    # Ensure we have Python integers for fixed dimensions
    num_points_int = int(num_points)
    max_num_neighbors_int = int(max_num_neighbors)
    
    def process_sample(sample_x):
        # Compute pairwise distances
        dists = jnp.sqrt(jnp.sum((sample_x[:, None, :] - sample_x[None, :, :]) ** 2, axis=-1))
        
        # Create mask for points within radius
        radius_mask = dists <= r
        
        # Handle self-loops
        if not loop:
            radius_mask = radius_mask & (jnp.arange(num_points_int)[:, None] != jnp.arange(num_points_int)[None, :])
        
        # Instead of using jnp.where, we'll convert the mask to a dense representation
        # Create a mesh grid of all possible indices
        row_indices = jnp.arange(num_points_int)[:, None].repeat(num_points_int, axis=1)
        col_indices = jnp.arange(num_points_int)[None, :].repeat(num_points_int, axis=0)
        
        # Get the flat indices where radius_mask is True
        flat_mask = radius_mask.flatten()
        flat_row_indices = row_indices.flatten()
        flat_col_indices = col_indices.flatten()
        
        # Use the mask to select valid indices
        # This approach keeps shapes static, unlike jnp.where
        valid_rows = jnp.where(flat_mask, flat_row_indices, -1)
        valid_cols = jnp.where(flat_mask, flat_col_indices, -1)
        
        # Sort by distance if needed
        if max_num_neighbors_int is not None:
            # Flatten distances for sorting
            flat_dists = dists.flatten()
            
            # Only consider valid distances (where mask is True)
            valid_dists = jnp.where(flat_mask, flat_dists, jnp.inf)
            
            # Get indices sorted by distance
            sorted_indices = jnp.argsort(valid_dists)
            
            # Take up to max_edges indices
            max_edges = num_points_int * max_num_neighbors_int
            sorted_indices = sorted_indices[:max_edges]
            
            # Use sorted indices to get rows and columns
            valid_rows = valid_rows[sorted_indices]
            valid_cols = valid_cols[sorted_indices]
        
        # Pad to a fixed size for batch processing
        max_edges = num_points_int * max_num_neighbors_int
        valid_rows = valid_rows[:max_edges]
        valid_cols = valid_cols[:max_edges]
        
        # Pad with -1 to indicate invalid edges
        padding = max_edges - len(valid_rows)
        if padding > 0:
            valid_rows = jnp.pad(valid_rows, (0, padding), constant_values=-1)
            valid_cols = jnp.pad(valid_cols, (0, padding), constant_values=-1)
        
        return jnp.stack([valid_rows, valid_cols])
    
    # Process each sample in the batch
    edge_indices = jax.vmap(process_sample)(x)
    
    return edge_indices


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
    radius: float
    max_degree: int
    input_dim: int
    hidden_dim: int
    ndim: int
    max_supernodes: int  # New parameter for fixed supernode count
    init_weights: str = "torch"
    
    @nn.compact
    def __call__(self, input_feat, input_pos, supernode_mask):
        """
        Args:
            input_feat: Shape (batch_size, num_points, input_dim)
            input_pos: Shape (batch_size, num_points, ndim)
            supernode_mask: Shape (batch_size, num_points) - Boolean mask for supernodes
            
        Returns:
            x: Shape (batch_size, max_supernodes, hidden_dim)
        """
        batch_size, num_points, _ = input_feat.shape
        
        # Radius graph - creates edges between nodes
        input_edges = batch_radius_graph_jax(
            x=input_pos,
            r=self.radius,
            max_num_neighbors=self.max_degree,
            loop=True,
        )  # Shape: (2, batch_size, num_edges)
        
        # Filter edges to only include supernode edges (where destination is a supernode)
        def filter_edges(edges, mask):
            # edges: (2, num_edges), mask: (num_points,)
            # Get indices of destination nodes (supernodes)
            dst_idx = edges[0]  # Shape: (num_edges,)
            
            # We need to handle invalid indices (-1) from the radius_graph results
            # Create a mask for valid indices
            valid_idx = dst_idx >= 0  # Shape: (num_edges,)
            
            # Safely get values from the mask using maximum to clip negative indices to 0
            # This avoids out-of-bounds indexing
            clipped_dst_idx = jnp.maximum(dst_idx, 0)  # Shape: (num_edges,)
            is_supernode = mask[clipped_dst_idx]  # Shape: (num_edges,)
            
            # Combine with valid_idx to ensure -1 indices are considered invalid regardless
            is_valid_supernode = is_supernode & valid_idx  # Shape: (num_edges,)
            
            # Add dimension for broadcasting with edges
            is_valid_supernode_expanded = is_valid_supernode[:, None]  # Shape: (num_edges, 1)
            
            # Filter edges - reshape for correct broadcasting
            filtered_src = jnp.where(is_valid_supernode, edges[0], -1)  # Shape: (num_edges,)
            filtered_dst = jnp.where(is_valid_supernode, edges[1], -1)  # Shape: (num_edges,)
            
            return jnp.stack([filtered_src, filtered_dst])
        
        # Apply filter to each batch
        input_edges = jax.vmap(filter_edges)(input_edges, supernode_mask)
        
        # Embed mesh
        input_proj = LinearProjection(features=self.hidden_dim, init_weights=self.init_weights)
        pos_embed = ContinuousSincosEmbed(dim=self.hidden_dim, ndim=self.ndim)
        
        x = input_proj(input_feat) + pos_embed(input_pos)
        
        # Create message inputs for each batch
        def create_messages(sample_x, sample_edges):
            # sample_x: (num_points, hidden_dim)
            # sample_edges: (2, max_edges)
            
            # Get valid edges (non-negative indices)
            # Using boolean indexing directly will fail during tracing
            all_src_idx = sample_edges[1]  # (max_edges,)
            all_dst_idx = sample_edges[0]  # (max_edges,)
            
            # Create mask for valid edges
            valid_mask = (all_dst_idx >= 0) & (all_src_idx >= 0)  # (max_edges,)
            
            # Convert negative indices to 0 (will be masked out later)
            safe_src_idx = jnp.maximum(all_src_idx, 0)  # (max_edges,)
            safe_dst_idx = jnp.maximum(all_dst_idx, 0)  # (max_edges,)
            
            # Get node features for source and destination
            # Handle out-of-bounds indices by clamping to valid range
            max_idx = sample_x.shape[0] - 1
            safe_src_idx = jnp.minimum(safe_src_idx, max_idx)
            safe_dst_idx = jnp.minimum(safe_dst_idx, max_idx)
            
            src_features = sample_x[safe_src_idx]  # (max_edges, hidden_dim)
            dst_features = sample_x[safe_dst_idx]  # (max_edges, hidden_dim)
            
            # Concatenate features
            message_features = jnp.concatenate([src_features, dst_features], axis=-1)  # (max_edges, 2*hidden_dim)
            
            # Apply mask - expand dimensions for broadcasting
            valid_mask_expanded = valid_mask[:, None]  # (max_edges, 1)
            masked_features = jnp.where(valid_mask_expanded, message_features, 0.0)  # (max_edges, 2*hidden_dim)
            
            # Create masked dst_idx for aggregation
            masked_dst_idx = jnp.where(valid_mask, all_dst_idx, -1)  # (max_edges,)
            
            return masked_features, masked_dst_idx
        
        # Apply to each batch
        message_inputs, dst_indices = jax.vmap(create_messages)(x, input_edges)
        # message_inputs: (batch_size, max_edges, 2*hidden_dim)
        # dst_indices: (batch_size, max_edges)
        
        # Message passing network
        message_net = nn.Sequential([
            LinearProjection(features=self.hidden_dim, init_weights=self.init_weights),
            lambda x: nn.gelu(x),
            LinearProjection(features=self.hidden_dim, init_weights=self.init_weights),
        ])
        
        # Process messages with message network
        processed_messages = jax.vmap(message_net)(message_inputs)
        # processed_messages: (batch_size, max_edges, hidden_dim)
        
        # Create mask for valid messages (based on dst_indices >= 0)
        valid_message_mask = (dst_indices >= 0).astype(jnp.float32)
        # valid_message_mask: (batch_size, max_edges)
        
        # Add dimension for broadcasting
        valid_message_mask = valid_message_mask[..., None]
        # valid_message_mask: (batch_size, max_edges, 1)
        
        # Apply mask to zero out invalid messages
        masked_messages = processed_messages * valid_message_mask
        # masked_messages: (batch_size, max_edges, hidden_dim)
        
        # Compute number of supernodes per batch - used for masking, not sizing
        num_supernodes = jnp.sum(supernode_mask, axis=1)
        
        # Use the fixed max_supernodes value instead of computing dynamically
        # Aggregate messages to supernodes
        x = batch_segment_aggregation(
            src=masked_messages,
            idx=dst_indices,
            num_segments=self.max_supernodes,  # Use fixed parameter
            reduce="mean"
        )
        
        # Create mask for valid supernodes using fixed max_supernodes
        supernode_valid_mask = jnp.arange(self.max_supernodes)[None, :] < num_supernodes[:, None]
        
        # Zero out invalid supernodes
        x = x * supernode_valid_mask[:, :, None]
        
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
            radius=self.radius,
            max_degree=self.max_degree,
            input_dim=self.input_dim,
            hidden_dim=self.gnn_dim,
            ndim=self.ndim,
            max_supernodes=self.max_supernodes,  # Pass the fixed parameter
        )
        
        # Encoder projection
        self.enc_proj = LinearProjection(
            features=self.enc_dim, 
            init_weights=self.init_weights, 
            optional=True
        )
        
        # Transformer blocks for conditioning
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
        
        # Perceiver pooling for conditioning
        if self.num_latent_tokens is not None:
            if self.cond_dim is None:
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
                self.perceiver = DitPerceiverPoolingBlock(
                    dim=self.perc_dim,
                    num_heads=self.perc_num_heads,
                    num_query_tokens=self.num_latent_tokens,
                    perceiver_kwargs=dict(
                        kv_dim=self.enc_dim,
                        cond_dim=self.cond_dim,
                        init_weights=self.init_weights,
                    ),
                )
        else:
            self.perceiver = None
            
        # Coordinate transformation components
        if self.output_coord_dim is not None:
            # Default coordinate encoder dimension if not provided
            if self.coord_enc_dim is None:
                self.coord_enc_dim = self.enc_dim
            
            # Combined feature and position encoder
            self.coord_input_proj = LinearProjection(
                features=self.coord_enc_dim,
                init_weights=self.init_weights
            )
            
            # Position embedding for coordinates
            self.coord_pos_embed = ContinuousSincosEmbed(
                dim=self.coord_enc_dim,
                ndim=self.ndim
            )
            
            # Transformer blocks for coordinate transformation
            self.coord_blocks = [
                DitBlock(
                    dim=self.coord_enc_dim, 
                    num_heads=self.coord_enc_num_heads, 
                    cond_dim=self.cond_dim,
                    init_weights=self.init_weights
                ) if self.cond_dim is not None else
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
    
    def __call__(self, input_feat_or_points, input_pos=None, supernode_mask=None, condition=None):
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
        # When using with UniversalAutoencoder in points-only mode,
        # make sure we use a valid supernode mask with correct count
        if input_pos is None:
            # We're in "points-only" mode
            points = input_feat_or_points
            input_feat = points
            input_pos = points
            
            # Create a mask where first max_supernodes points are supernodes
            # This ensures a fixed number of supernodes compatible with our model
            batch_size, num_points = points.shape[0], points.shape[1]
            supernode_mask = jnp.zeros((batch_size, num_points), dtype=bool)
            # valid_supernodes = jnp.minimum(num_points, self.max_supernodes)
            supernode_mask = supernode_mask.at[:, :self.max_supernodes].set(True)
        else:
            # Standard mode with separate feature and position tensors
            input_feat = input_feat_or_points
        
        # Check inputs
        assert input_feat.ndim == 3, "expected tensor (batch_size, num_points, input_dim)"
        assert input_pos.ndim == 3, "expected tensor (batch_size, num_points, ndim)"
        assert input_feat.shape[0] == input_pos.shape[0], "batch dimensions must match"
        assert input_feat.shape[1] == input_pos.shape[1], "number of points must match"
        assert supernode_mask.ndim == 2, "supernode_mask should be a 2D boolean tensor (batch_size, num_points)"
        assert supernode_mask.shape[0] == input_feat.shape[0], "batch dimensions must match"
        assert supernode_mask.shape[1] == input_feat.shape[1], "number of points must match"
        
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"
            assert condition.shape[0] == input_feat.shape[0], "batch dimensions must match"
        
        # ----- Conditioning branch -----
        # Supernode pooling
        x = self.supernode_pooling(
            input_feat=input_feat,
            input_pos=input_pos,
            supernode_mask=supernode_mask,
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