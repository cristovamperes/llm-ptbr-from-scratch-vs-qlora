#!/usr/bin/env bash
set -e
source tcc_venv/bin/activate
python - <<'PY'
import tensorflow as tf
print('version', tf.__version__)
print('gpu devices', tf.config.list_physical_devices('GPU'))
x = tf.zeros([1, 10, 32])
layer = tf.keras.layers.LSTM(64)
y = layer(x)
print('output shape', y.shape)
print('backend', layer.cell.__class__.__name__)
PY