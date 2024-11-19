import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from N_mc2data import MCReader, Block
from tensorflow.keras import regularizers
import gc
import random

from tensorflow.keras import layers, models, backend as K

def generate_continuous_noise(batch_size, chunk_shape, seed):
    # Set the seed for reproducibility
    tf.random.set_seed(seed)
    # Generate continuous noise in range [0, 5]
    continuous_noise = tf.random.uniform(shape=(batch_size, *chunk_shape), minval=0.0, maxval=5.0)

    return continuous_noise

def generate_discrete_noise(batch_size, chunk_shape, seed):
    # Set the seed for reproducibility
    tf.random.set_seed(seed)
    # Generate random integers in range [0, 5]
    discrete_noise = tf.random.uniform(shape=(batch_size, *chunk_shape), minval=0, maxval=6, dtype=tf.int32)

    return discrete_noise

krn_size = 4
activ = 'leaky_relu'
# Denoise unet - and hope :)
# Process with convolutional layers
def Conv3D(x, filters, krn_size = 3, strides_shape = (1,1,1), normalize = True):
    x = layers.Conv3D(filters, kernel_size=krn_size, padding='same', strides=strides_shape, activation=activ)(x)
    if normalize:
        x = layers.BatchNormalization()(x)
    return x
def TConv3D(x, filters, krn_size = 3, strides_shape = (1,1,1), normalize = True):
    x = layers.Conv3DTranspose(filters, kernel_size=krn_size, padding='same', strides=strides_shape, activation=activ)(x)
    if normalize:
        x = layers.BatchNormalization()(x)
    return x
def upsample(x, factor=2):
    return layers.UpSampling3D(size=(factor, factor, factor))(x)

# apply conditions
def attention_thing(bottleneck_features, guidance_value):
    # Reshape the guidance_value to match the shape of the bottleneck features for dot product calculation
    guidance_reshaped = layers.Reshape((1, 1, 1, 1))(guidance_value)
    # Step 1: Compute the similarity between guidance and each bottleneck feature (dot product)
    similarity = layers.Multiply()([bottleneck_features, guidance_reshaped])
    # Step 2: Apply softmax over the last dimension (the 256 feature dimension) to get attention weights
    attention_weights = layers.Softmax(axis=-1)(similarity)
    # Step 3: Use the attention weights to scale the bottleneck features (weighted sum)
    attended_features = layers.Multiply()([bottleneck_features, attention_weights])
    return attended_features

def cave_attention(x, cave_map, depth, use_cave_attention):
    #cave_map = upsample(cave_map, depth)
    #cave_map = Conv3D(cave_map, 1, 1, (depth,depth,depth), False)
    cave_map = layers.Conv3D(1, kernel_size=1, padding='same', strides=(depth,depth,depth))(cave_map)
    '''
    similarity = layers.Multiply()([x, cave_map])
    attention_weights = layers.Softmax(axis=-1)(similarity)
    attended_features = layers.Multiply()([x, attention_weights])
    '''
    # Compute attention weights
    #attention_weights = tf.nn.sigmoid(cave_map)
    attention_weights = layers.Activation('sigmoid')(cave_map)

    # Apply attention weights to the feature map
    attended_features = layers.Multiply()([x, attention_weights])
    #return attended_features
    #return use_cave_attention * attended_features + (1 - use_cave_attention) * x
    return attended_features

class ConditionalCaveMapLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(ConditionalCaveMapLayer, self).__init__(**kwargs)
        # Define layers in __init__
        self.conv2d = layers.Conv2D(320, kernel_size=3, strides=(1, 1), padding='same')
        self.permute = layers.Permute((1, 3, 2))
        self.reshape = layers.Reshape((16, 320, 16, 1))
        self.conv3d1 = layers.Conv3D(32, kernel_size=3, padding='same', activation='leaky_relu')
        self.conv3d2 = layers.Conv3D(64, kernel_size=3, padding='same', activation='leaky_relu')
        self.conv3d3 = layers.Conv3D(128, kernel_size=3, padding='same', activation='leaky_relu')
        self.conv3d = layers.Conv3D(1, kernel_size=3, padding='same', activation='sigmoid')

    def call(self, inputs):
        input_heightmap, input_cave_density, use_cave_attention = inputs

        # Create cave map
        reshaped_heightmap = layers.Reshape((16, 16, 1))(input_heightmap)
        cave_init = self.conv2d(reshaped_heightmap)
        cave_init = layers.Multiply()([cave_init, input_cave_density])

        cave_init_reshaped = self.permute(cave_init)
        x = self.reshape(cave_init_reshaped)
        x = self.conv3d1(x)
        x = self.conv3d2(x)
        x = self.conv3d3(x)
        cave_map_3d = self.conv3d(x)

        return cave_map_3d


def build_generator():
    # Inputs
    init_chunks_indices = layers.Input(shape=(16, 320, 16, 6), name='init_chunks')  # (None, 16, 16)
    input_heightmap = layers.Input(shape=(16, 16), name='input_heightmap')  # (None, 16, 16)
    input_cave_density = layers.Input(shape=(1,), name='input_cave_density')  # (None, 1)
    use_cave_attention = layers.Input(shape=(), dtype=tf.float32, name='use_cave_attention')

    cave_map_3d = ConditionalCaveMapLayer()([input_heightmap, input_cave_density, use_cave_attention])

    # Embbeding chunk
    #embedding_layer = tf.keras.layers.Embedding(input_dim=6, output_dim=8) # 6.. hm?
    #embedded_chunk = embedding_layer(init_chunks_indices) # (_, 16, 320, 16, 8)
    #print(np.shape(embedded_chunk))

    embedded_chunk = init_chunks_indices

    # INPUT
    # 8, 16x320

    # depth 0
    #  conv 3x3 64
    d0_conv_1 = Conv3D(embedded_chunk, 64, 3, (1,1,1), False) # 64, 16 x 320
    #  conv 3x3 64
    d0_conv_2 = Conv3D(d0_conv_1, 64, 3, (1,1,1)) # 64, 16 x 320
    #  conv 3x3 64 max-pool
    d0_conv_3 = Conv3D(d0_conv_2, 64, 3, (1,1,1)) # 64, 16 x 320
    d0_conv_3 = cave_attention(d0_conv_3, cave_map_3d, 1, use_cave_attention)

    # depth 1
    #  conv 3x3 128
    d1_conv_1 = Conv3D(d0_conv_3, 128, 3, (2,2,2)) # 128, 8 x 160
    d1_conv_2 = Conv3D(d1_conv_1, 128, 3, (1,1,1)) # 128, 8 x 160
    d1_conv_3 = Conv3D(d1_conv_2, 128, 3, (1,1,1)) # 128, 8 x 160
    d1_conv_3 = cave_attention(d1_conv_3, cave_map_3d, 2, use_cave_attention)

    # depth 2
    d2_conv_1 = Conv3D(d1_conv_3, 256, 3, (2,2,2)) # 256, 4 x 80
    d2_conv_2 = Conv3D(d2_conv_1, 256, 3, (1,1,1)) # 256, 4 x 80
    d2_conv_3 = Conv3D(d2_conv_2, 256, 3, (1,1,1)) # 256, 4 x 80
    d2_conv_3 = cave_attention(d2_conv_3, cave_map_3d, 4, use_cave_attention)

    # depth 3 (bottleneck)
    d3_conv_1 = Conv3D(d2_conv_3, 512, 3, (2,2,2)) # 512, 2 x 40
    d3_conv_2 = Conv3D(d3_conv_1, 512, 3, (1,1,1)) # 512, 2 x 40
    d3_conv_3 = Conv3D(d3_conv_2, 512, 3, (1,1,1)) # 512, 2 x 40

    d3_conv_3 = cave_attention(d3_conv_3, cave_map_3d, 8, use_cave_attention)


    # depth 2
    print(np.shape(d3_conv_3))
    d2_up = upsample(d3_conv_3)                    # 512, 4 x 80
    print(np.shape(d2_up))
    d2_tconv_half = TConv3D(d2_up, 256, 3, (1,1,1))# 256, 4 x 80
    d2_tconv_full = layers.Concatenate()([d2_conv_3, d2_tconv_half]) # 512, 4 x 80
    d2_tconv_1 = TConv3D(d2_tconv_full, 256, 3, (1,1,1)) # 256, 4 x 80
    d2_tconv_2 = TConv3D(d2_tconv_1, 256, 3, (1,1,1)) # 256, 4 x 80

    # depth 1
    d1_up = upsample(d2_tconv_2)                   # 256, 8 x 160
    d1_tconv_half = TConv3D(d1_up, 128, 3, (1,1,1))# 128, 8 x 160
    d1_tconv_full = layers.Concatenate()([d1_conv_3, d1_tconv_half]) # 256, 8 x 160
    d1_tconv_1 = TConv3D(d1_tconv_full, 128, 3, (1,1,1)) # 128, 8 x 160
    d1_tconv_2 = TConv3D(d1_tconv_1, 128, 3, (1,1,1)) # 128, 8 x 160

    # depth 0
    d0_up = upsample(d1_tconv_2)                   # 128, 16 x 320
    d0_tconv_half = TConv3D(d0_up, 64, 3, (1,1,1)) # 64, 16 x 320
    d0_tconv_full = layers.Concatenate()([d0_conv_3, d0_tconv_half]) # 128, 16 x 320
    d0_tconv_1 = TConv3D(d0_tconv_full, 64, 3, (1,1,1)) # 64, 16 x 320
    d0_tconv_2 = TConv3D(d0_tconv_1, 64, 3, (1,1,1)) # 64, 16 x 320

    # OUTPUT
    output = layers.Conv3DTranspose(6, kernel_size=1, padding='same', strides=(1, 1, 1), activation='softmax')(d0_tconv_2) # 6, 16 x 320


    # Build and return the model
    generator_model = models.Model(inputs=[init_chunks_indices, input_heightmap, input_cave_density, use_cave_attention],
                                   outputs=[output, cave_map_3d])
    return generator_model


# STRIDES, is a kernel shift on reading




def build_discriminator(heightmap_shape=(16, 16, 1), cave_density_shape=(1,), chunk_shape=(16, 320, 16, 6)):
    # Inputs
    input_chunk = layers.Input(shape=chunk_shape)
    #input_heightmap = layers.Input(shape=heightmap_shape)
    #input_cave_density = layers.Input(shape=cave_density_shape)

    # Embbeding chunk
    #embedding_layer = tf.keras.layers.Embedding(input_dim=6, output_dim=8) # 6.. hm?
    #embedded_chunk = embedding_layer(input_chunk)

    # On first layer there is used kernel of size 5 with a hope to better capture huge caves

    # depth 0
    #  conv 3x3 64
    d0_conv_1 = Conv3D(input_chunk, 64, 5, (1,1,1), False) # 64, 16 x 320
    #  conv 3x3 64
    d0_conv_2 = Conv3D(d0_conv_1, 64, 3, (1,1,1)) # 64, 16 x 320
    #  conv 3x3 64 max-pool
    d0_conv_3 = Conv3D(d0_conv_2, 64, 3, (1,1,1)) # 64, 16 x 320
    #d0_conv_3 = cave_attention(d0_conv_3, cave_map_3d, 1)

    # depth 1
    #  conv 3x3 128
    d1_conv_1 = Conv3D(d0_conv_3, 128, 3, (2,2,2)) # 128, 8 x 160
    d1_conv_2 = Conv3D(d1_conv_1, 128, 3, (1,1,1)) # 128, 8 x 160
    d1_conv_3 = Conv3D(d1_conv_2, 128, 3, (1,1,1)) # 128, 8 x 160
    #d1_conv_3 = cave_attention(d1_conv_3, cave_map_3d, 2)

    # depth 2
    d2_conv_1 = Conv3D(d1_conv_3, 256, 3, (2,2,2)) # 256, 4 x 80
    d2_conv_2 = Conv3D(d2_conv_1, 256, 3, (1,1,1)) # 256, 4 x 80
    d2_conv_3 = Conv3D(d2_conv_2, 256, 3, (1,1,1)) # 256, 4 x 80
    #d2_conv_3 = cave_attention(d2_conv_3, cave_map_3d, 4)

    # depth 3 (bottleneck)
    d3_conv_1 = Conv3D(d2_conv_3, 512, 3, (2,2,2)) # 512, 2 x 40
    d3_conv_2 = Conv3D(d3_conv_1, 512, 3, (1,1,1)) # 512, 2 x 40
    d3_conv_3 = Conv3D(d3_conv_2, 512, 3, (1,1,1)) # 512, 2 x 40

    #d3_conv_3 = cave_attention(d3_conv_3, cave_map_3d, 8)

    d2_up = upsample(d3_conv_3)                    # 512, 4 x 80
    print(np.shape(d2_up))
    d2_tconv_half = TConv3D(d2_up, 256, 3, (1,1,1))# 256, 4 x 80
    d2_tconv_full = layers.Concatenate()([d2_conv_3, d2_tconv_half]) # 512, 4 x 80
    d2_tconv_1 = TConv3D(d2_tconv_full, 256, 3, (1,1,1)) # 256, 4 x 80
    d2_tconv_2 = TConv3D(d2_tconv_1, 256, 3, (1,1,1)) # 256, 4 x 80

    # depth 1
    d1_up = upsample(d2_tconv_2)                   # 256, 8 x 160
    d1_tconv_half = TConv3D(d1_up, 128, 3, (1,1,1))# 128, 8 x 160
    d1_tconv_full = layers.Concatenate()([d1_conv_3, d1_tconv_half]) # 256, 8 x 160
    d1_tconv_1 = TConv3D(d1_tconv_full, 128, 3, (1,1,1)) # 128, 8 x 160
    d1_tconv_2 = TConv3D(d1_tconv_1, 128, 3, (1,1,1)) # 128, 8 x 160

    # depth 0
    d0_up = upsample(d1_tconv_2)                   # 128, 16 x 320
    d0_tconv_half = TConv3D(d0_up, 64, 3, (1,1,1)) # 64, 16 x 320
    d0_tconv_full = layers.Concatenate()([d0_conv_3, d0_tconv_half]) # 128, 16 x 320
    d0_tconv_1 = TConv3D(d0_tconv_full, 64, 3, (1,1,1)) # 64, 16 x 320
    d0_tconv_2 = TConv3D(d0_tconv_1, 64, 3, (1,1,1)) # 64, 16 x 320

    output = layers.Conv3D(1, kernel_size=1, strides=(1,1,1), padding='same', activation='sigmoid')(d0_tconv_2)


    # Build model
    discriminator_model = models.Model(inputs=[input_chunk], outputs=output)
    return discriminator_model

generator = None
discriminator = None
cave_scale = tf.Variable(2.0, trainable=True, dtype=tf.float32)

def build():
    print("[ChunkGAN] Building models...")
    global generator
    global discriminator
    generator = build_generator()
    discriminator = build_discriminator()
    print("[ChunkGAN] Two models builded!")

def load_model(model_name):
    print("[ChunkGAN] Loading models...")
    from tensorflow.keras.models import load_model
    generator.load_weights(f'{model_name}.gen.h5')
    discriminator.load_weights(f'{model_name}.dis.h5')
    print(f"[ChunkGAN] Model '{model_name}' loaded!")

def apply_noise(chunks, seed = None):
    batch_size = tf.shape(chunks)[0]
    if seed is None:
        seed = random.randint(1, 100)
    noise_intensity = 0.5
    noise = generate_discrete_noise(batch_size, (16, 320, 16), seed)
    noised_chunks = (1 - noise_intensity) * chunks + noise_intensity * tf.cast(noise, tf.float32)
    noised_chunks = tf.round(noised_chunks)  # Round to nearest integer
    noised_chunks = tf.clip_by_value(noised_chunks, 0, 5)
    return noised_chunks

# Inputs:
# - amount:                         uint        -> 1...N
# - heightmaps:                     (N, 16, 16) -> 0...320
# - cave_dense:                     (N, 1)      -> 0...32256
# - (optional) seed:                uint        -> any (trained on 0...100)
# - (optional) include_cave_map:    Bool        -> True/False
def generate(amount, heightmaps, cave_dens, seed = None):
    print("[ChunkGAN] Generating chunks...")
    IN_cave_dens = cave_dens * cave_scale
    IN_cave_dens = tf.reshape(IN_cave_dens, (amount, 1))

    use_cave_attention = tf.convert_to_tensor([1.] * amount, dtype=tf.float32)
    use_cave_attention = tf.reshape(use_cave_attention, (amount, 1))

    IN_heightmaps = tf.convert_to_tensor(heightmaps, dtype=tf.float32) / 320.0
    IN_heightmaps = tf.reshape(IN_heightmaps, (amount, 16, 16))

    # prepare
    volume = np.zeros(shape=(amount, 16, 320, 16), dtype=np.float32)
    stone_id = 3
    air_id = 0
    for d in range(amount):
        for x in range(16):
            for z in range(16):
                volume[d][x, 0:int(heightmaps[d][x, z]), z] = stone_id
                volume[d][x, int(heightmaps[d][x, z]):320, z] = air_id

    noised_chunk = apply_noise(tf.reshape(volume, (amount, 16, 320, 16)), seed)
    IN_noised_chunk_one_hot = tf.one_hot(tf.cast(noised_chunk, tf.uint8), depth=6)

    chunks, cave_maps = generator.predict([IN_noised_chunk_one_hot, IN_heightmaps, IN_cave_dens, use_cave_attention], batch_size=1)

    print(f"[ChunkGAN] Generated {amount} chunks!")
    
    return chunks, cave_maps
