import numpy as np
import tensorflow as tf

betas = tf.linspace(start=0.0001, stop=0.02, num=300)
alphas = 1.0 - betas
alphas_cumprod = tf.math.cumprod(alphas, axis=0)
alphas_cumprod_prev = tf.pad(alphas_cumprod[:-1], [(1, 0),], constant_values=1.0)
sqrt_recip_alphas = tf.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = tf.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def diffuse_sample(sample, t, ):
    noise = tf.random.normal(sample.shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, sample.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, sample.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t * sample + sqrt_one_minus_alphas_cumprod_t * noise, noise


def get_index_from_list(vals, t, x_shape):
    batch_size = tf.shape(t)[0]
    out = tf.gather(vals, axis=-1)
    return tf.reshape(out, [batch_size, (len(x_shape)-1)])

