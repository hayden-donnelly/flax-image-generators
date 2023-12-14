from typing import Any, Callable, List

import jax
import jax.numpy as jnp
import flax.linen as nn

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_max_frequency:float
    embedding_min_frequency:float = 1.0
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(self.embedding_min_frequency),
                jnp.log(self.embedding_max_frequency),
                self.embedding_dim // 2,
                dtype=self.dtype
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
            axis=-1,
            dtype=self.dtype
        )
        return embeddings

class ConvResBlock(nn.Module):
    num_features: int
    num_groups: int
    kernel_size: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, time_emb=None):
        input_features = x.shape[-1]
        if input_features == self.num_features:
            residual = x
        else:
            residual = nn.Conv(self.num_features, kernel_size=(1, 1))(x)
        x = nn.Conv(self.num_features, kernel_size=(self.kernel_size, self.kernel_size))(x)
        x = nn.GroupNorm(self.num_groups)(x)
        x = self.activation_fn(x)
        if time_emb is not None:
            time_emb = nn.Dense(self.num_features)(time_emb)
            time_emb = self.activation_fn(time_emb)
            time_emb = jnp.broadcast_to(time_emb, x.shape)
            x = x + time_emb
        x = nn.Conv(self.num_features, kernel_size=(self.kernel_size, self.kernel_size))(x)
        x = nn.GroupNorm(self.num_groups)(x)
        x = self.activation_fn(x)
        x = x + residual
        return x

class ConvDownBlock(nn.Module):
    num_features:int
    num_groups:int
    block_depth:int
    pool_size:int
    pool_stride:int
    activation_fn:Callable

    @nn.compact
    def __call__(self, x, time_emb, skips):
        for _ in range(self.block_depth):
            x = ConvResBlock(
                num_features=self.num_features, 
                num_groups=self.num_groups,
                patch_size=self.patch_size,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                activation_fn=self.activation_fn
            )(x, time_emb)
            skips.append(x)
        x = nn.max_pool(
            inputs=x, 
            window_shape=(self.pool_size, self.pool_size), 
            strides=(self.pool_stride, self.pool_stride)
        )
        return x, skips
    
class ConvUpBlock(nn.Module):
    num_features: int
    num_groups: int
    block_depth: int
    upsample_factor: int
    upsample_method: str
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, time_emb, skips):
        upsample_shape = (
            x.shape[0], 
            x.shape[1] * self.factor, 
            x.shape[2] * self.factor, 
            x.shape[3]
        )
        x = jax.image.resize(x, upsample_shape, method=self.upsample_method)

        for _ in range(self.block_depth):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ConvResBlock(
                num_features=self.num_features,
                num_groups=self.num_groups,
                patch_size=self.patch_size,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                activation_fn=self.activation_fn
            )(x, time_emb)
        return x, skips
    
class ConvDiffusion(nn.Module):
    embedding_dim: int
    embedding_max_frequency: float
    num_features: List[int]
    num_groups: List[int]
    block_depth: int
    channels_out: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x, diffusion_time):
        assert len(self.num_features) == len(self.num_groups), (
            'num_features must be the same length as num_groups, got',
            f'{len(self.num_features)} and {len(self.num_groups)}.'
        )

        time_emb = SinusoidalEmbedding(
            embedding_dim=self.embedding_dim,
            embedding_max_frequency=self.embedding_max_frequency
        )(diffusion_time)

        skips = []
        for i in range(len(self.num_features)-1):
            x, skips = ConvDownBlock(
                num_features=self.num_features[i],
                num_groups=self.num_groups[i],
                block_depth=self.block_depth,
                activation_fn=self.activation_fn
            )(x, time_emb, skips)
        for _ in range(self.block_depth):
            x = ConvResBlock(
                num_features=self.num_features[-1],
                num_groups=self.num_groups[-1],
                patch_size=self.patch_sizes[-1],
                head_dim=self.head_dims[-1],
                num_heads=self.head_dims[-1],
                activation_fn=self.activation_fn
            )(x, time_emb)
        for i in range(len(self.num_features)-1, -1, -1):
            x, skips = ConvUpBlock(
                num_features=self.num_features[i],
                num_groups=self.num_features[i],
                block_depth=self.block_depth,
                activation_fn=self.activation_fn
            )(x, time_emb, skips)

        x = nn.Conv(self.channels_out, kernel_size=(1, 1))(x)
        return x