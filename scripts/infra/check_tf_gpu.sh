#!/usr/bin/env bash
source tcc_venv/bin/activate
python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print([d.name for d in device_lib.list_local_devices()])
PY