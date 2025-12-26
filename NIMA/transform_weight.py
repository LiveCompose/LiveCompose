import argparse, numpy as np, tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import tf2onnx

def build_keras_nima(weights_path: str):
    base = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling="avg", weights=None)
    x = Dropout(0.75)(base.output)
    x = Dense(10, activation="softmax")(x)
    model = Model(base.input, x)
    model.load_weights(weights_path)
    return model

def export_onnx(h5_path: str, onnx_path: str):
    model = build_keras_nima(h5_path)
    input_name = model.inputs[0].name.split(":")[0]
    spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name=input_name),)
    model.summary()
    tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13, output_path=onnx_path,
        inputs_as_nchw=[input_name]   
    )
    print(f"✅ Exported ONNX: {onnx_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--onnx", required=True)
    args = ap.parse_args()
    export_onnx(args.h5, args.onnx)

    # 在根目录运行示例：
    #python ./LiveCompose/NIMA/transform_weight.py   --h5 ./LiveCompose/NIMA/weights/inception_resnet_weights.h5   --onnx ./LiveCompose/NIMA/weights/nima_inception_resnet.onnx