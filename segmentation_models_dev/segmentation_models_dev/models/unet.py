from keras_applications import get_submodules_from_kwargs

from ._common_blocks import Conv2dBn, Conv2dGn
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones
import tensorflow_addons as tfa

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, drop_rate=None, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            drop_rate=drop_rate,
            **kwargs
        )(input_tensor)

    return wrapper

def Conv3x3GnReLU(filters, use_groupnorm, groupnorm_groups, drop_rate=None, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dGn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_groupnorm=use_groupnorm,
            groupnorm_groups=groupnorm_groups,
            name=name,
            drop_rate=drop_rate,
            **kwargs
        )(input_tensor)

    return wrapper

def DecoderUpsamplingX2Block(filters, stage, drop_rate=None, use_groupnorm=False, groupnorm_groups=2, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if use_batchnorm == True and use_groupnorm == True:
        raise ValueError('Cannot use both BatchNorm and GroupNorm')

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        if use_groupnorm:
            x = Conv3x3GnReLU(filters, use_groupnorm, groupnorm_groups, drop_rate=drop_rate, name=conv1_name)(x)
            x = Conv3x3GnReLU(filters, use_groupnorm, groupnorm_groups, drop_rate=drop_rate, name=conv2_name)(x)
        else:
            # this conditional branch includes not using batchnorm, by passing corresponding flag
            x = Conv3x3BnReLU(filters, use_batchnorm, drop_rate=drop_rate, name=conv1_name)(x)
            x = Conv3x3BnReLU(filters, use_batchnorm, drop_rate=drop_rate, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, drop_rate=None, use_groupnorm=False, groupnorm_groups=2, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    gn_name = 'decoder_state{}a_gn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = gn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not (use_batchnorm or use_groupnorm),
        )(input_tensor)

        if use_batchnorm == True and use_groupnorm == True:
            raise ValueError('Cannot use both BatchNorm and GroupNorm')

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if use_groupnorm:
            x = tfa.layers.GroupNormalization(groups=groupnorm_groups, axis=gn_axis, name=gn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        if use_groupnorm:
            x = Conv3x3GnReLU(filters, use_groupnorm, groupnorm_groups, drop_rate, name=conv_block_name)(x)
        else:
            x = Conv3x3BnReLU(filters, use_batchnorm, drop_rate=drop_rate, name=conv_block_name)(x)

        return x

    return layer


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        use_groupnorm=False,
        groupnorm_groups=2,
        drop_rate=None
):
    if use_batchnorm == True and use_groupnorm == True:
        raise ValueError('Cannot use both BatchNorm and GroupNorm')

    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        if use_groupnorm:
            x = Conv3x3GnReLU(512, use_groupnorm, groupnorm_groups, drop_rate=drop_rate, name='center_block1')(x)
            x = Conv3x3GnReLU(512, use_groupnorm, groupnorm_groups, drop_rate=drop_rate, name='center_block2')(x)
        else:
            x = Conv3x3BnReLU(512, use_batchnorm, drop_rate=drop_rate, name='center_block1')(x)
            x = Conv3x3BnReLU(512, use_batchnorm, drop_rate=drop_rate, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None
        if use_groupnorm:
            x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm, drop_rate=drop_rate, 
                                use_groupnorm=use_groupnorm, groupnorm_groups=groupnorm_groups)(x, skip)
        else:
            x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm, drop_rate=drop_rate)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def Unet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        decoder_use_groupnorm=False,
        decoder_groupnorm_groups=2,
        decoder_drop_rate=None,
        encoder_activation='relu',
        **kwargs
):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.

    Returns:
        ``keras.models.Model``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """
    if decoder_use_batchnorm == True and decoder_use_groupnorm == True:
        raise ValueError('Cannot use both BatchNorm and GroupNorm')

    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    # backwards compatability with original models not using resnet18_modified
    if backbone_name == 'resnet18_modified':
        print("using resnet18_modified")
        backbone = Backbones.get_backbone(
            backbone_name,
            input_shape=input_shape,
            # weights=encoder_weights,
            weights=None,
            include_top=False,
            encoder_activation=encoder_activation,
            **kwargs,
        )
    else:
        print("using original backbones")
        backbone = Backbones.get_backbone(
            backbone_name,
            input_shape=input_shape,
            weights=encoder_weights,
            include_top=False,
            **kwargs,
        )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
        use_groupnorm=decoder_use_groupnorm,
        groupnorm_groups=decoder_groupnorm_groups,
        drop_rate=decoder_drop_rate
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
