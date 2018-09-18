import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

#整個網路變化的維度：784->256->28->256->784
batch_size = 100
#原始輸入維度，28*28=784
original_dim = 784
#編碼後的編碼的維度
latent_dim = 28
#中間隱藏層的維度
intermediate_dim = 256
#迭代50次
epochs = 50
#初始化時的標準差
epsilon_std = 1.0

#編碼器的結構
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
# 平均向量(mean vector)
z_mean = Dense(latent_dim)(h)
# 標準差向量(standard deviation vector)
z_log_var = Dense(latent_dim)(h)

#使用平均向量（mean vector）和標準差向量（standard deviation vector）合成隱向量
def sampling(args):
    z_mean, z_log_var = args
    #使用標準正態分佈初始化
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,stddev=epsilon_std)
    #合成公式
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
#z即為所要求得的隱含變量
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
# 解碼器的結構
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
#x_decoded_mean 即為解碼器輸出的結果
x_decoded_mean = decoder_mean(h_decoded)

# Custom loss layer
#自定義損失層，損失包含兩個部分：圖片的重構誤差（均方差Square Loss）以及隱變量與單位高斯分割之間的差異（KL-散度KL-Divergence Loss）。
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)#Square Loss
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)#KL-Divergence Loss
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

#將損失層加入網路
y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

# train the VAE on MNIST digits
#使用MNIST數據集進行訓練
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#圖像數據歸一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#將圖像數據轉換為784維的向量
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#模型訓練設置
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))
# build a model to project inputs on the latent space
#編碼器的網絡結構，將輸入圖形映射為代碼，即隱含變量
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
#將所有測試集中的圖片通過編碼器轉換為隱含變量（二維變量），並將其在二維空間中進行繪圖
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()