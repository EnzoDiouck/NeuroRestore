"""
Custom Keras layers for VAE model
Adapted from image_reconstructor_VAE.ipynb
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResNetBlock(layers.Layer):
    """Residual block for VAE encoder/decoder"""
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        # First convolution
        self.conv1 = layers.Conv2D(
            self.filters, 
            self.kernel_size, 
            padding='same',
            use_bias=True  # Changed from False to match saved model
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        # Second convolution
        self.conv2 = layers.Conv2D(
            self.filters,
            self.kernel_size,
            padding='same',
            use_bias=True  # Changed from False to match saved model
        )
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut connection (if dimensions don't match)
        if input_shape[-1] != self.filters:
            self.shortcut = layers.Conv2D(
                self.filters,
                1,
                padding='same',
                use_bias=True  # Changed from False to match saved model
            )
        else:
            self.shortcut = None
        
        self.relu2 = layers.ReLU()
        super(ResNetBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
        
        x = layers.add([x, shortcut])
        x = self.relu2(x)
        
        return x
    
    def get_config(self):
        config = super(ResNetBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SubPixelConv2D(layers.Layer):
    """
    Sub-Pixel Convolution (Pixel Shuffle) for upsampling
    
    More effective than transposed convolution, reduces checkerboard artifacts.
    Rearranges elements from depth into spatial dimensions.
    """
    def __init__(self, scale=2, filters=None, **kwargs):
        super(SubPixelConv2D, self).__init__(**kwargs)
        self.scale = scale
        self.filters = filters
    
    def build(self, input_shape):
        # Conv to expand channels for pixel shuffle
        # Output channels = filters * scale^2
        out_channels = (self.filters if self.filters else input_shape[-1]) * (self.scale ** 2)
        self.conv = layers.Conv2D(out_channels, 3, padding='same')
        super().build(input_shape)
    
    def call(self, inputs):
        x = self.conv(inputs)
        # Pixel shuffle: (B, H, W, C*r^2) -> (B, H*r, W*r, C)
        return tf.nn.depth_to_space(x, self.scale)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'scale': self.scale,
            'filters': self.filters
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UpsampleBlock(layers.Layer):
    """
    Upsampling block using UpSampling2D + Conv2D + SubPixelConv2D
    
    Combines multiple upsampling strategies for better quality
    """
    def __init__(self, filters, use_subpixel=True, **kwargs):
        super(UpsampleBlock, self).__init__(**kwargs)
        self.filters = filters
        self.use_subpixel = use_subpixel
    
    def build(self, input_shape):
        if self.use_subpixel:
            # SubPixel convolution path
            self.subpixel = SubPixelConv2D(scale=2, filters=self.filters)
        else:
            # Traditional upsampling path
            self.upsample = layers.UpSampling2D(size=2, interpolation='bilinear')
            self.conv = layers.Conv2D(self.filters, 3, padding='same')
        
        self.bn = layers.BatchNormalization()
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        if self.use_subpixel:
            x = self.subpixel(inputs)
        else:
            x = self.upsample(inputs)
            x = self.conv(x)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'use_subpixel': self.use_subpixel
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
