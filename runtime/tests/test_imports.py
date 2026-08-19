def test_numpy():
    import numpy  # noqa: F401


def test_pandas():
    import pandas  # noqa: F401


def test_polars():
    import polars  # noqa: F401


def test_loguru():
    import loguru  # noqa: F401


def test_tqdm():
    import tqdm  # noqa: F401


def test_mutagen():
    import mutagen  # noqa: F401


def test_soundfile():
    import soundfile  # noqa: F401


def test_librosa():
    import librosa  # noqa: F401


def test_sacrebleu():
    import sacrebleu  # noqa: F401


def test_torch_cuda_available():
    import torch

    assert torch.cuda.is_available(), "torch cannot see a CUDA device"


def test_torchaudio():
    import torchaudio  # noqa: F401


def test_torchcodec():
    import torchcodec  # noqa: F401


def test_transformers():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # noqa: F401


def test_accelerate():
    import accelerate  # noqa: F401


def test_peft():
    import peft  # noqa: F401


def test_vllm():
    import vllm  # noqa: F401


def test_tensorflow_gpu_available():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "tensorflow cannot see a CUDA device"


def test_keras_gpu_op():
    import keras

    x = keras.ops.ones((8, 8))
    y = keras.ops.matmul(x, x)
    assert float(keras.ops.sum(y)) == 512.0
