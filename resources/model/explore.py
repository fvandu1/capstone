from keras.models import load_model
from keras.datasets import mnist
import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0],28,28,1)
y_test = keras.utils.to_categorical(y_test)
x_test = x_test.astype('float32')
x_test /= 255

x_train = x_train.astype('float32')
x_train /= 255

model = load_model('mnist.keras')

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


def plotSamples():
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.axis("off")
        for j in range(10):
            if K.eval(y_test[i])[j] == 1:
                plt.title(j)
        plt.imshow(K.eval(x_test)[i])
    plt.show()


def plotPixelDistro():
    counts, bins = np.histogram(x_test)
    plt.stairs(counts, bins)
    plt.title("Pixel distribution")
    plt.show()


def plotPCA():
    pca = PCA(n_components=2)
    proj = pca.fit_transform(x_train.reshape(-1,784))
    plt.figure(figsize=(15,7))
    plt.title('PCA Visualization')
    plt.scatter(proj[:,0], proj[:,1], c=y_train, cmap="Paired")
    plt.ylim([-8,8])
    plt.colorbar(ticks=range(10))
    plt.show()


def plotTSNE():
    embeddings = TSNE().fit_transform(x_train.reshape(-1,784))
    plt.figure(figsize=(15,7))
    plt.title('t-SNE')
    plt.scatter(embeddings[:,0],embeddings[:,1],c=y_train, cmap=plt.cm.get_cmap("jet",10),marker='.')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5,9.5)
    plt.show()

        