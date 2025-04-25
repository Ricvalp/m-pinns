
class SupernodePooling(nn.Module):
    def __init__(
            self,
            radius,
            max_degree,
            input_dim,
            hidden_dim,
            ndim,
            init_weights="torch",
    ):
        super().__init__()
        self.radius = radius
        self.max_degree = max_degree
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights

        self.input_proj = LinearProjection(input_dim, hidden_dim, init_weights=init_weights)
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        self.message = nn.Sequential(
            LinearProjection(hidden_dim * 2, hidden_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(hidden_dim, hidden_dim, init_weights=init_weights),
        )
        self.output_dim = hidden_dim

    def forward(self, input_feat, input_pos, supernode_idxs, batch_idx):
        assert input_feat.ndim == 2
        assert input_pos.ndim == 2
        assert supernode_idxs.ndim == 1

        # radius graph
        input_edges = radius_graph(
            x=input_pos,
            r=self.radius,
            max_num_neighbors=self.max_degree,
            batch=batch_idx,
            loop=True,
            # inverted flow direction is required to have sorted dst_indices
            flow="target_to_source",
        )
        is_supernode_edge = torch.isin(input_edges[0], supernode_idxs)
        input_edges = input_edges[:, is_supernode_edge]

        # embed mesh
        x = self.input_proj(input_feat) + self.pos_embed(input_pos)

        # create message input
        dst_idx, src_idx = input_edges.unbind()
        x = torch.concat([x[src_idx], x[dst_idx]], dim=1)
        x = self.message(x)
        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = dst_idx.unique(return_counts=True)
        # first index has to be 0
        # NOTE: padding for target indices that dont occour is not needed as self-loop is always present
        padded_counts = torch.zeros(len(counts) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[1:] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")

        # sanity check: dst_indices has len of batch_size * num_supernodes and has to be divisible by batch_size
        # if num_supernodes is not set in dataset this assertion fails
        batch_size = batch_idx.max() + 1
        assert dst_indices.numel() % batch_size == 0

        # convert to dense tensor (dim last)
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )

        return x

class EncoderSupernodes(nn.Module):
    def __init__(
            self,
            input_dim,
            ndim,
            radius,
            max_degree,
            gnn_dim,
            enc_dim,
            enc_depth,
            enc_num_heads,
            perc_dim=None,
            perc_num_heads=None,
            num_latent_tokens=None,
            cond_dim=None,
            init_weights="truncnormal",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim
        self.radius = radius
        self.max_degree = max_degree
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.num_latent_tokens = num_latent_tokens
        self.condition_dim = cond_dim
        self.init_weights = init_weights

        # supernode pooling
        self.supernode_pooling = SupernodePooling(
            radius=radius,
            max_degree=max_degree,
            input_dim=input_dim,
            hidden_dim=gnn_dim,
            ndim=ndim,
        )

        # blocks
        self.enc_proj = LinearProjection(gnn_dim, enc_dim, init_weights=init_weights, optional=True)
        if cond_dim is None:
            block_ctor = PrenormBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(dim=enc_dim, num_heads=enc_num_heads, init_weights=init_weights)
                for _ in range(enc_depth)
            ],
        )

        # perceiver pooling
        if num_latent_tokens is None:
            self.perceiver = None
        else:
            if cond_dim is None:
                block_ctor = partial(
                    PerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        init_weights=init_weights,
                    ),
                )
            else:
                block_ctor = partial(
                    DitPerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        cond_dim=cond_dim,
                        init_weights=init_weights,
                    ),
                )
            self.perceiver = block_ctor(
                dim=perc_dim,
                num_heads=perc_num_heads,
                num_query_tokens=num_latent_tokens,
            )

    def forward(self, input_feat, input_pos, supernode_idxs, batch_idx, condition=None):
        # check inputs
        assert input_feat.ndim == 2, "expected sparse tensor (batch_size * num_inputs, input_dim)"
        assert input_pos.ndim == 2, "expected sparse tensor (batch_size * num_inputs, ndim)"
        assert len(input_feat) == len(input_pos), "expected input_feat and input_pos to have same length"
        assert supernode_idxs.ndim == 1, "supernode_idxs is a 1D tensor of indices that are used as supernodes"
        assert batch_idx.ndim == 1, f"batch_idx should be 1D tensor that assigns elements of the input to samples"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # supernode pooling
        x = self.supernode_pooling(
            input_feat=input_feat,
            input_pos=input_pos,
            supernode_idxs=supernode_idxs,
            batch_idx=batch_idx,
        )

        # project to encoder dimension
        x = self.enc_proj(x)

        # transformer
        x = self.blocks(x, **cond_kwargs)

        # perceiver
        if self.perceiver is not None:
            x = self.perceiver(kv=x, **cond_kwargs)

        return x