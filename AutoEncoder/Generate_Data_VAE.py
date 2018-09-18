import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

#Read the training file path
file_dict = {
    'Train':'Processed_Data_Train.csv',
    'Test':'Processed_Data_Test.csv',
    'Minus21':'Processed_Data_Minus21.csv'
}
train_data = pd.read_csv(file_dict['Train'])
test_data = pd.read_csv(file_dict['Test'])
minus_data = pd.read_csv(file_dict['Minus21'])

def parse_fl(data):
    # This function mainly to transfer the raw csv file to 
    # training features and labels. 
    
    # Select the features in the input data
    features = data[:, 0:122]
    # Select the labels in the input data
    labels = data[:, 122]
    #trans the label from text to

    return features, labels

train_data = train_data.values
test_data = test_data.values
minus_data = minus_data.values

# Prepare the training features and labels 
train_Features, train_Labels = parse_fl(train_data)
test_Features, test_Labels = parse_fl(test_data)
minus_Features, minus_Labels = parse_fl(minus_data)


#整個網路變化的維度：122->61->30->61->61
batch_size = 100
#原始輸入維度，122
original_dim = 122
#編碼後的編碼的維度
latent_dim = 2
#中間隱藏層的維度
intermediate_dim = 20
#迭代50次
epochs = 10
#初始化時的標準差
epsilon_std = 0.1

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

'''
# train the VAE on MNIST digits
#使用MNIST數據集進行訓練
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#圖像數據歸一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#將圖像數據轉換為784維的向量
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
'''
x_train = train_Features
x_test = test_Features
y_train = train_Labels
y_test = test_Labels

#模型訓練設置
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        #batch_size=batch_size,
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

# build a digit generator that can sample from the learned distribution
#製作一個解碼器，用來將隱藏向量解碼出圖片
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
#绘制一个15个图像*15个图像的图
n = 15  # figure with 15x15 digits
#每个图像的大小为28*28
digit_size = 28
#初始化为0
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
# 生成因变量空间（二维）中的数据，数据满足高斯分布。这些数据构成隐变量，用于图像的生成。
#ppf为累积分布函数（cdf）的反函数，累积分布函数是概率密度函数（pdf）的积分。np.linspace(0.05, 0.95, n)为累计分布函数的输出值（y值），现在我们需要其对应的x值，所以使用cdf的反函数，这些x值构成隐变量。
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])#add by weihao: 1*2
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)#add by weihao: the generated image
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()