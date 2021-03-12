import cv2
import os
from tqdm import tqdm
from keras.layers import BatchNormalization, Reshape, Dense, Input, LeakyReLU, Conv2D, Conv2DTranspose, Concatenate, ReLU, Dropout, ZeroPadding2D
from keras.models import Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import numpy as np
import time


def load_img(file_path):
    img = cv2.imread(file_path)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    img = (img/127.5) - 1
    return img


train_a = []
train_b = []

trainA_path = r'C:\Users\Shad Humydee\Desktop\ATL\01. Noise Removal\R & D\_assets\apple2orange\trainA'

for files in tqdm(os.listdir(trainA_path)):
    file_path = os.path.join(trainA_path, files)
    #print(file_path)
    input_img = load_img(file_path)
    train_a.append(input_img)

trainB_path = r'C:\Users\Shad Humydee\Desktop\ATL\01. Noise Removal\R & D\_assets\apple2orange\trainB'

for files in tqdm(os.listdir(trainB_path)):
    file_path = os.path.join(trainB_path, files)
    input_img = load_img(file_path)
    train_b.append(input_img)

train_a = np.array(train_a)
train_b = np.array(train_b)


def generator():
    image_input = Input(shape=(256, 256, 3))

    # Encoder Network

    conv_1 = Conv2D(64, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                    padding='same')(image_input)
    act_1 = LeakyReLU(alpha=0.2)(conv_1)

    conv_2 = Conv2D(128, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                    padding='same')(act_1)
    batch_norm_2 = BatchNormalization(momentum=0.8)(conv_2)
    act_2 = LeakyReLU(alpha=0.2)(batch_norm_2)

    conv_3 = Conv2D(256, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                    padding='same')(act_2)
    batch_norm_3 = BatchNormalization(momentum=0.8)(conv_3)
    act_3 = LeakyReLU(alpha=0.2)(batch_norm_3)

    conv_4 = Conv2D(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                    padding='same')(act_3)
    batch_norm_4 = BatchNormalization(momentum=0.8)(conv_4)
    act_4 = LeakyReLU(alpha=0.2)(batch_norm_4)

    conv_5 = Conv2D(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                    padding='same')(act_4)
    batch_norm_5 = BatchNormalization(momentum=0.8)(conv_5)
    act_5 = LeakyReLU(alpha=0.2)(batch_norm_5)

    conv_6 = Conv2D(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                    padding='same')(act_5)
    batch_norm_6 = BatchNormalization(momentum=0.8)(conv_6)
    act_6 = LeakyReLU(alpha=0.2)(batch_norm_6)

    conv_7 = Conv2D(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                    padding='same')(act_6)
    batch_norm_7 = BatchNormalization()(conv_7)
    act_7 = LeakyReLU(alpha=0.2)(batch_norm_7)

    conv_8 = Conv2D(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                    padding='same')(act_7)
    batch_norm_8 = BatchNormalization(momentum=0.8)(conv_8)
    act_8 = LeakyReLU(alpha=0.2)(batch_norm_8)

    # Decoder Network and skip connections with encoder

    convt_1 = Conv2DTranspose(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                              padding='same')(act_8)
    batch_normt_1 = BatchNormalization(momentum=0.8)(convt_1)
    drop_1 = Dropout(0.5)(batch_normt_1)
    actt_1 = ReLU()(drop_1)
    concat_1 = Concatenate()([actt_1, act_7])

    convt_2 = Conv2DTranspose(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                              padding='same')(concat_1)
    batch_normt_2 = BatchNormalization(momentum=0.8)(convt_2)
    drop_2 = Dropout(0.5)(batch_normt_2)
    actt_2 = ReLU()(drop_2)
    concat_2 = Concatenate()([actt_2, act_6])

    convt_3 = Conv2DTranspose(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                              padding='same')(concat_2)
    batch_normt_3 = BatchNormalization(momentum=0.8)(convt_3)
    drop_3 = Dropout(0.5)(batch_normt_3)
    actt_3 = ReLU()(drop_3)
    concat_3 = Concatenate()([actt_3, act_5])

    convt_4 = Conv2DTranspose(512, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                              padding='same')(concat_3)
    batch_normt_4 = BatchNormalization(momentum=0.8)(convt_4)
    actt_4 = ReLU()(batch_normt_4)
    concat_4 = Concatenate()([actt_4, act_4])

    convt_5 = Conv2DTranspose(256, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                              padding='same')(concat_4)
    batch_normt_5 = BatchNormalization(momentum=0.8)(convt_5)
    actt_5 = ReLU()(batch_normt_5)
    concat_5 = Concatenate()([actt_5, act_3])

    convt_6 = Conv2DTranspose(128, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                              padding='same')(concat_5)
    batch_normt_6 = BatchNormalization(momentum=0.8)(convt_6)
    actt_6 = ReLU()(batch_normt_6)
    concat_6 = Concatenate()([actt_6, act_2])

    convt_7 = Conv2DTranspose(64, 4, strides=2, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02),
                              padding='same')(concat_6)
    batch_normt_7 = BatchNormalization(momentum=0.8)(convt_7)
    actt_7 = ReLU()(batch_normt_7)
    concat_7 = Concatenate()([actt_7, act_1])

    outputs = Conv2DTranspose(3, 4, strides=2, use_bias=False, activation='tanh',
                              kernel_initializer=RandomNormal(mean=0., stddev=0.02), padding='same')(concat_7)
    gen_model = Model(image_input, outputs)

    #     gen_model.summary()

    return gen_model


genA = generator()
genB = generator()


def discriminator():

    img_inp = Input(shape = (256, 256, 3))

    conv_1 = Conv2D(64,4,strides=2,use_bias=False,kernel_initializer=RandomNormal(mean=0.,stddev=0.02),padding='same')(img_inp)
    act_1 = LeakyReLU(alpha=0.2)(conv_1)

    conv_2 = Conv2D(128,4,strides=2,use_bias=False,kernel_initializer=RandomNormal(mean=0.,stddev=0.02),padding='same')(act_1)
    batch_norm_2 = BatchNormalization(momentum=0.8)(conv_2)
    act_2 = LeakyReLU(alpha=0.2)(batch_norm_2)

    conv_3 = Conv2D(256,4,strides=2,use_bias=False,kernel_initializer=RandomNormal(mean=0.,stddev=0.02),padding='same')(act_2)
    batch_norm_3 = BatchNormalization(momentum=0.8)(conv_3)
    act_3 = LeakyReLU(alpha=0.2)(batch_norm_3)

    zero_pad = ZeroPadding2D()(act_3)

    conv_4 = Conv2D(512,4,strides=1,use_bias=False,kernel_initializer=RandomNormal(mean=0.,stddev=0.02))(zero_pad)
    batch_norm_4 = BatchNormalization(momentum=0.8)(conv_4)
    act_4 = LeakyReLU(alpha=0.2)(batch_norm_4)

    zero_pad_1 = ZeroPadding2D()(act_4)
    outputs = Conv2D(1,4,strides=1,use_bias=False,kernel_initializer=RandomNormal(mean=0.,stddev=0.02))(zero_pad_1)

    disc_model = Model(img_inp, outputs)

#     disc_model.summary()
    return disc_model


discA = discriminator()
discB = discriminator()
#discA.summary()


def combined():
    inputA = Input(shape=(256, 256, 3))
    inputB = Input(shape=(256, 256, 3))
    gen_imgB = genA(inputA)
    gen_imgA = genB(inputB)

    # for cycle consistency
    reconstruct_imgA = genB(gen_imgB)
    reconstruct_imgB = genA(gen_imgA)

    # identity mapping
    gen_orig_imgB = genA(inputB)
    gen_orig_imgA = genB(inputA)

    discA.trainable = False
    discB.trainable = False

    valid_imgA = discA(gen_imgA)
    valid_imgB = discA(gen_imgB)

    comb_model = Model([inputA, inputB],
                       [valid_imgA, valid_imgB, reconstruct_imgA, reconstruct_imgB, gen_orig_imgA, gen_orig_imgA])
    #     comb_model.summary()
    return comb_model


comb_model = combined()

optimizer = Adam(0.0002, 0.5)

discA.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
discB.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
comb_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae','mae'],loss_weights=[  1, 1, 10, 10, 1, 1],optimizer=optimizer)

disc_patch = (30, 30, 1)
epochs = 200
valid = np.ones((1,) + disc_patch)
fake = np.zeros((1,) + disc_patch)


def train():
    for j in range(epochs):
        t1 = time.time()
        for i in range(len(train_a)):
            img_a = np.expand_dims(train_a[i], axis=0)
            img_b = np.expand_dims(train_b[i], axis=0)
            img_b_gen = genA.predict(img_a)
            img_a_gen = genB.predict(img_b)

            # train discriminator A
            dA_real_loss = discA.train_on_batch(img_a, valid)
            dA_fake_loss = discA.train_on_batch(img_a_gen, fake)

            # train discriminator B
            dB_real_loss = discB.train_on_batch(img_b, valid)
            dB_fake_loss = discB.train_on_batch(img_b_gen, fake)

            # train generator
            g_loss = comb_model.train_on_batch([img_a, img_b], [valid, valid, img_a, img_b, img_a, img_b])
            if i == 993:
                print('time taken for one epoch', time.time() - t1)
                print(j, i, dA_real_loss, dA_fake_loss, dB_real_loss, dB_fake_loss, g_loss)


train()