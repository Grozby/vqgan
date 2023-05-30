from collections import namedtuple
from typing import List

import tensorflow as tf


class LPIPS(tf.keras.models.Model):
    """
    Learned Perceptual Image Patch Similarity

    References:
        https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf
    """

    def __init__(self, weights_path: str, **kwargs):
        super().__init__(**kwargs)
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.vgg = VGG16()
        self.linear_layers = [
            NetLinLayer(name=f"netlin{i}") for i in range(len(self.channels))
        ]

        self.trainable = False

    def build(self, input_shape):
        self.load_weights(self.weights_path)

    def call(self, x, **kwargs):
        real_x, fake_x = x
        features_real = self.vgg(self.scaling_layer(real_x))
        features_fake = self.vgg(self.scaling_layer(fake_x))
        diffs = [(norm_tensor(fr) - norm_tensor(ff))**2
                 for fr, ff in zip(features_real, features_fake)]

        return tf.reduce_sum([
            spatial_average(ll(d)) for ll, d in zip(self.linear_layers, diffs)
        ])


class ScalingLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = tf.Variable(
            tf.constant([
                -.030,
                -.088,
                -.188,
            ])[None, None, None, :],
            trainable=False,
            name="shift",
        )
        self.scale = tf.Variable(
            tf.constant([
                .458,
                .448,
                .450,
            ])[None, None, None, :],
            trainable=False,
            name="scale",
        )

    def call(self, x, **kwargs):
        return (x - self.shift) / self.scale


class NetLinLayer(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.block = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=1,
                strides=1,
                use_bias=False,
            )
        ])

    def call(self, x, **kwargs):
        return self.block(x)


class VGG16(tf.keras.Model):

    def __init__(self):
        super().__init__()

        vgg_pretrained_features = tf.keras.applications.VGG16(
            include_top=True,
            weights=None,
        ).layers

        self.slice1 = tf.keras.models.Sequential(
            self._add_zero_padding(vgg_pretrained_features[0:3]),
            name="features1",
        )
        self.slice2 = tf.keras.models.Sequential(
            self._add_zero_padding(vgg_pretrained_features[3:6]),
            name="features2",
        )
        self.slice3 = tf.keras.models.Sequential(
            self._add_zero_padding(vgg_pretrained_features[6:10]),
            name="features3",
        )
        self.slice4 = tf.keras.models.Sequential(
            self._add_zero_padding(vgg_pretrained_features[10:14]),
            name="features4",
        )
        self.slice5 = tf.keras.models.Sequential(
            self._add_zero_padding(vgg_pretrained_features[14:18]),
            name="features5",
        )

        self.trainable = False

    def _add_zero_padding(self, layers: List) -> List:
        new_layers = [layers[0]]
        for ll in layers[1:]:
            new_layers.append(tf.keras.layers.ZeroPadding2D(padding=(1, 1)))
            assert isinstance(ll, tf.keras.layers.Conv2D)
            ll.padding = "valid"
            new_layers.append(ll)
        return new_layers

    def call(self, x, **kwargs):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        vgg_outputs = namedtuple(
            "VGGOutputs",
            [
                "relu1_2",
                "relu2_2",
                "relu3_3",
                "relu4_3",
                "relu5_3",
            ],
        )
        return vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


def norm_tensor(x):
    norm_factor = tf.math.sqrt(tf.reduce_sum(x**2, axis=3, keepdims=True))
    return x / (norm_factor + 1e-10)


def spatial_average(x):
    return tf.reduce_mean(x, axis=[1, 2], keepdims=True)
