from keras.models import load_model
import tensorflow as tf
import tensorflow.keras.layers as L
# from keras import models 
# from efficientnet.tfkeras import EfficientNetB4
import efficientnet.tfkeras as efn
import cv2
import numpy as np



freq_pos, freq_neg = ([0.02551686, 0.02356118, 0.11966847, 0.00181598, 0.17992177,
       0.05252375, 0.05620227, 0.10178804, 0.04763457, 0.03208232,
       0.01280499, 0.01438815, 0.02127957, 0.04176755]), ([0.97448314, 0.97643882, 0.88033153, 0.99818402, 0.82007823,
       0.94747625, 0.94379773, 0.89821196, 0.95236543, 0.96791768,
       0.98719501, 0.98561185, 0.97872043, 0.95823245])

pos_weights = freq_neg
neg_weights = freq_pos

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):

    def weighted_loss(y_true, y_pred):

        # initialize loss to zero
        loss = 0.0
        
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss_pos = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += loss_pos + loss_neg
        return loss

    return weighted_loss

IMAGE_SIZE=[128, 128]

model = tf.keras.Sequential([
    efn.EfficientNetB1(
        input_shape=(*IMAGE_SIZE, 3),
        weights='imagenet',
        include_top=False),
    L.GlobalAveragePooling2D(),
    L.Dense(1024, activation = 'relu'), 
    L.Dense(14, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam( learning_rate=1e-4, amsgrad=False), 
    #loss = 'binary_crossentropy',
    loss = get_weighted_loss(pos_weights, neg_weights),
    metrics = ['accuracy']
)

model.load_weights('efficent_net_b1_trained_weights.h5')



img = cv2.imread("chest1.png")
# print(type(img))
# print(img.shape)
img = cv2.resize(img,(128,128))
img = img.reshape((1,128,128,3))
img= np.array(img)
img= img.astype(np.float32)
# print(img)

# # model = load_model('mymodel1.h5')
# # model.summary()

prediction = model.predict(img)
print(prediction)



