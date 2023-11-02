#Authors: Piotr Gazda, Patryk Lipka, Sara Witek
#Semester VI, ISMIP1

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda, LeakyReLU, concatenate
from yolo_functions import parse_annotation, ImageReader, normalize,\
SimpleBatchGenerator, space_to_depth_x2, custom_loss, OutputRescaler, find_high_class_probability_bbox, \
nonmax_suppression, draw_boxes

# Tensorflow GPU memory usage setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#Parameters definition
LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']
ANCHORS = np.array([0.08285376, 0.13705531,
                       0.20850361, 0.39420716,
                       0.80552421, 0.77665105,
                       0.42194719, 0.62385487])
GRID_H,  GRID_W  = 13 , 13
ANCHORS[::2]     = ANCHORS[::2]*GRID_W
ANCHORS[1::2]    = ANCHORS[1::2]*GRID_H
IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE       = 2
TRUE_BOX_BUFFER  = 50
BOX = int(len(ANCHORS)/2)
CLASS = len(LABELS)
generator_config = {
    'IMAGE_H'         : IMAGE_H,
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}
LR_SCHEDULE = [
    (0, 0.01),
    (35, 0.001),
    (60, 0.0001),
]
EPOCHS = 50

train_image_folder = "F:\GazdaWork\datasets\VOC\VOCdevkit\VOC2012\JPEGImages\\"
train_annot_folder = "F:\GazdaWork\datasets\VOC\VOCdevkit\VOC2012\Annotations\\"

#VOC dataset handling
train_image, seen_train_labels = parse_annotation(train_annot_folder,train_image_folder, labels=LABELS)

#Creating training batch generator
train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                             norm=normalize, shuffle=True)


#Neural network definition
input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))


x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D(pool_size=(2, 2))(x)


x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)


x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)


x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)


x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)


x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)


x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)


x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)


x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)


x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)


skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])


x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = tf.keras.activations.sigmoid(x)

x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([input_image, true_boxes], output)
model.summary()

layer   = model.layers[-4]
weights = layer.get_weights()

new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

layer.set_weights([new_kernel, new_bias])

#Training preparation
checkpoint = ModelCheckpoint('F:\GazdaWork\yolo_v2_models\checkpoints_2.09\\',
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1)


optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss ,optimizer=optimizer, metrics=['accuracy'])

tf.config.experimental_run_functions_eagerly(True)

#Training and savinge the model
history = model.fit_generator(generator        = train_batch_generator,
                    steps_per_epoch  = len(train_batch_generator),
                    epochs           = EPOCHS,
                    verbose          = 1,
                    callbacks        = [checkpoint],
                    max_queue_size   = 3)

MODEL_2 = "F:\GazdaWork\yolo_v2_models\model_2.09\\"
model.save(MODEL_2)

epochs_range = range(EPOCHS)

#Conversion and saving to TensorflowLite format
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_2)

#Quantize model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

MODEL_2_TFLITE = "F:\GazdaWork\TFLite_Models\yolo_v2\\2021-09-02_2_\model.tflite"

with open(MODEL_2_TFLITE, 'wb') as f:
  f.write(tflite_model)


#Reading the saved model and using TensorflowLite to run interference on image
interpreter = tf.lite.Interpreter(MODEL_2_TFLITE)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


IMAGE_FOR_INTERFERENCE = "F:\GazdaWork\datasets\VOC\VOCdevkit\VOC2007\JPEGImages\\003260.jpg"
imageReader = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=lambda image : image / 255.)
out = imageReader.fit(IMAGE_FOR_INTERFERENCE)
X_test = np.expand_dims(out,0).astype('float32')
dummy_array = np.ones((1,1,1,1,TRUE_BOX_BUFFER,4)).astype('float32')


interpreter.set_tensor(input_details[0]['index'], dummy_array)
interpreter.set_tensor(input_details[1]['index'], X_test)
interpreter.invoke()
y_pred = interpreter.get_tensor(output_details[0]['index'])


netout         = y_pred[0]
outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
netout_scale   = outputRescaler.fit(netout)


#Finding bounding boxes with high predicted class probability
obj_threshold = 0.03
boxes = find_high_class_probability_bbox(netout_scale,obj_threshold)

#Applying non-max suppression to distinguish only one bounding box for ground truth box
iou_threshold = 0.01
final_boxes = nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)
if len(final_boxes) > 0:
    print("{} final number of boxes".format(len(final_boxes)))

#Displaying results
ima = draw_boxes(X_test[0],final_boxes,LABELS,verbose=True)
figsize = (15,15)
plt.figure(figsize=figsize)
plt.imshow(ima)
plt.show()