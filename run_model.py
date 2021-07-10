import edward2
import uncertainty_baselines
from uncertainty_baselines.models.resnet50_batchensemble import resnet50_batchensemble
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds
ds = tfds.load('mnist', split='train', shuffle_files=True)
ds = ds.shuffle(1024).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
for example in ds.take(1):
  image, label = example["image"], example["label"]

model = resnet50_batchensemble(input_shape=(28, 28, 1), ensemble_size=50, num_classes=10, random_sign_init=42, use_ensemble_bn=True)
model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])
history = model.fit(x=image)
