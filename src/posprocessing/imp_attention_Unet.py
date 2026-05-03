# https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831

# b. Implementation in Keras
def expend_as(tensor, rep):
    
    # Anonymous lambda function to expand the specified axis by a factor of argument, rep.
    # If tensor has shape (512,512,N), lambda will return a tensor of shape (512,512,N*rep), if specified axis=2

    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = Conv3D(filters=inter_shape, kernel_size=1, strides=1, padding='same')(g)

    # Getting the x signal to the same shape as the gating signal
    theta_x = Conv3D(filters=inter_shape, kernel_size=3, strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2], shape_x[3] // shape_g[3]), padding='same')(x)

    # Element-wise addition of the gating and x signals
    add_xg = add([phi_g, theta_x])
    add_xg = Activation('relu')(add_xg)

    # 1x1x1 convolution
    psi = Conv3D(filters=1, kernel_size=1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(psi)

    # Upsampling psi back to the original dimensions of x signal
    upsample_sigmoid_xg = UpSampling3D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(psi)

    # Expanding the filter axis to the number of filters in the original x signal
    upsample_sigmoid_xg = expend_as(upsample_sigmoid_xg, shape_x[4])

    # Element-wise multiplication of attention coefficients back onto original x signal
    attn_coefficients = multiply([upsample_sigmoid_xg, x])

    # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
    output = Conv3D(filters=shape_x[4], kernel_size=1, strides=1, padding='same')(attn_coefficients)
    output = BatchNormalization()(output)
    return output




# tradicional implementation 
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

def conv_block(input, num_filters):
    conv_output = Conv2D(num_filters, (3, 3), padding='same')(input)
    conv_output = BatchNormalization()(conv_output)
    conv_output = Activation('relu')(conv_output)

    conv_output = Conv2D(num_filters, (3, 3), padding='same')(conv_output)
    conv_output = BatchNormalization()(conv_output)
    conv_output = Activation('relu')(conv_output)

    return conv_output

def encoder_block(input, num_filters):
    conv_output = conv_block(input, num_filters)
    pooling_output = MaxPooling2D((2,2), strides=(2, 2))(conv_output)

    return conv_output, pooling_output

def decoder_block(input_tensor, skip_features, num_filters):
    transposed = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input_tensor)
    concat_result = Concatenate()([transposed, skip_features])
    conv_output = conv_block(concat_result, num_filters)
    return conv_output

def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    final_layer = Conv2D(num_classes, 1, padding='same', activation='sigmoid')
    outputs = final_layer(d4)

    model = Model(inputs, outputs, name='U-Net')
    return model

# Função para calcular o Índice de Jaccard
def jaccard_index(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / union


def get_model():
    numberBnd = len(indexBands)
    my_model = build_unet(input_shape=(256,256, numberBnd), num_classes=1)
    print(my_model.summary())
    my_model.compile(
        optimizer= Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics= varMetrics
    )
    return my_model