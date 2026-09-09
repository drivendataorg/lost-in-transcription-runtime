"""Tests for ctranslate2 and faster-whisper.

Split out from test_imports.py because the GPU test builds a model, and that
takes more code than an import test.
"""


N_MELS = 80
D_MODEL = 8
FFN_DIM = 16
VOCAB_SIZE = 64
ENCODER_FRAMES = 1500
DECODER_POSITIONS = 448

# The four tokens a Whisper prompt must open with. ctranslate2 rejects a
# prompt that does not start with <|startoftranscript|>.
WHISPER_PROMPT = [
    "<|startoftranscript|>",
    "<|es|>",
    "<|transcribe|>",
    "<|notimestamps|>",
]

def test_ctranslate2():
    """Check that ctranslate2 imports and counts the GPU.

    This test cannot see a broken GPU path. ctranslate2 opens cuBLAS by name
    at the first computation on the GPU, not at import, so nothing here
    reaches that code. test_ctranslate2_whisper_gpu_decode covers it.
    """
    import ctranslate2

    assert ctranslate2.get_cuda_device_count() >= 1, (
        "ctranslate2 cannot see a CUDA device"
    )


def test_faster_whisper():
    from faster_whisper import WhisperModel  # noqa: F401


def _tiny_whisper_model(path):
    """Write a one-layer Whisper model with random weights in CT2 format.

    Built here so the test needs no checkpoint and no network. The container
    has neither. The weights are random, so the output text means nothing. The
    point is to run the kernels: the encoder convolutions and the attention
    and feed-forward layers all reach cuBLAS.
    """
    import numpy as np
    from ctranslate2.specs.whisper_spec import WhisperSpec

    rng = np.random.default_rng(0)

    def weights(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.02

    def fill_norm(layer_norm):
        layer_norm.gamma = np.ones(D_MODEL, np.float32)
        layer_norm.beta = np.zeros(D_MODEL, np.float32)

    def fill_attention(attention, fused_widths):
        """Fill one attention block.

        Self-attention packs query, key and value into a single linear, so its
        first weight is three times as wide. Cross-attention keeps the query
        apart and packs key and value together.
        """
        fill_norm(attention.layer_norm)
        for index, width in enumerate(fused_widths):
            attention.linear[index].weight = weights(D_MODEL * width, D_MODEL)
            attention.linear[index].bias = np.zeros(D_MODEL * width, np.float32)
        attention.linear[-1].weight = weights(D_MODEL, D_MODEL)
        attention.linear[-1].bias = np.zeros(D_MODEL, np.float32)

    def fill_ffn(ffn):
        fill_norm(ffn.layer_norm)
        ffn.linear_0.weight = weights(FFN_DIM, D_MODEL)
        ffn.linear_0.bias = np.zeros(FFN_DIM, np.float32)
        ffn.linear_1.weight = weights(D_MODEL, FFN_DIM)
        ffn.linear_1.bias = np.zeros(D_MODEL, np.float32)

    spec = WhisperSpec(1, 1, 1, 1)

    encoder = spec.encoder
    encoder.conv1.weight = weights(D_MODEL, N_MELS, 3)
    encoder.conv1.bias = np.zeros(D_MODEL, np.float32)
    encoder.conv2.weight = weights(D_MODEL, D_MODEL, 3)
    encoder.conv2.bias = np.zeros(D_MODEL, np.float32)
    encoder.position_encodings.encodings = weights(ENCODER_FRAMES, D_MODEL)
    fill_norm(encoder.layer_norm)
    fill_attention(encoder.layer[0].self_attention, [3])
    fill_ffn(encoder.layer[0].ffn)

    decoder = spec.decoder
    decoder.embeddings.weight = weights(VOCAB_SIZE, D_MODEL)
    decoder.position_encodings.encodings = weights(DECODER_POSITIONS, D_MODEL)
    fill_norm(decoder.layer_norm)
    decoder.projection.weight = weights(VOCAB_SIZE, D_MODEL)
    fill_attention(decoder.layer[0].self_attention, [3])
    fill_attention(decoder.layer[0].attention, [1, 2])
    fill_ffn(decoder.layer[0].ffn)

    spec.register_vocabulary(
        WHISPER_PROMPT
        + ["<|endoftext|>"]
        + [f"tok{i}" for i in range(VOCAB_SIZE - len(WHISPER_PROMPT) - 1)]
    )
    spec.validate()

    path.mkdir(parents=True, exist_ok=True)
    spec.save(str(path))
    return path

def test_ctranslate2_whisper_gpu_decode(tmp_path):
    """Run a Whisper model on the GPU to check that ctranslate2 finds CUDA.

    An import test cannot see this break. libctranslate2 does not link cuBLAS.
    It opens the soname libcublas.so.12 the first time a model computes on the
    GPU. That file lives under site-packages, which the loader does not
    search, so the Dockerfile adds the directory to the loader cache. Without
    that the import tests still pass and this test aborts the process.
    """
    import numpy as np

    import ctranslate2

    model_dir = _tiny_whisper_model(tmp_path / "tiny-whisper")
    model = ctranslate2.models.Whisper(
        str(model_dir), device="cuda", compute_type="float16"
    )
    mel = ctranslate2.StorageView.from_array(
        np.zeros((1, N_MELS, ENCODER_FRAMES * 2), np.float32)
    )
    result = model.generate(model.encode(mel), [WHISPER_PROMPT], max_length=5)
    assert result[0].sequences[0], "ctranslate2 returned no tokens on the GPU"
