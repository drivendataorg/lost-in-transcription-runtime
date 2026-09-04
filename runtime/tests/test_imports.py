def test_numpy():
    import numpy  # noqa: F401


NUMPY_CAP_OVERRIDDEN = {
    # pyctcdecode declares numpy<2.0 but only calls numpy APIs that still exist
    # in numpy 2.x, so the runtime raises its cap on purpose.
    "pyctcdecode",
}


def test_numpy_satisfies_declared_bounds():
    """Check that the installed numpy fits every installed package's numpy range.

    An override that names numpy without naming a package rewrites the numpy
    requirement of every package, including the upper bounds that numba and
    mistral-common declare. uv resolves past those bounds and reports no
    conflict, so this test is the only place the mismatch shows up.
    """
    from importlib.metadata import distributions

    import numpy
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    installed = Version(numpy.__version__)
    violations = []
    for dist in distributions():
        name = dist.metadata["Name"]
        if not name or canonicalize_name(name) in NUMPY_CAP_OVERRIDDEN:
            continue
        for raw in dist.requires or []:
            req = Requirement(raw)
            if canonicalize_name(req.name) != "numpy":
                continue
            # Skip a requirement that an extra or a Python version gates off.
            if req.marker is not None and not req.marker.evaluate():
                continue
            if not req.specifier.contains(installed, prereleases=True):
                violations.append(f"{canonicalize_name(name)} needs numpy{req.specifier}")

    assert not violations, f"numpy {installed} is out of range for: " + ", ".join(
        sorted(violations)
    )


def test_numpy_override_names_a_package():
    """Check that every numpy override in pyproject.toml names one package.

    An override entry written as a plain string applies to all packages. Scope
    a numpy override to the single package whose cap is wrong. Then the caps
    other packages declare stay in force.
    """
    import tomllib
    from pathlib import Path

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text())
    overrides = config.get("tool", {}).get("uv", {}).get("override-dependencies", [])

    unscoped = [
        entry
        for entry in overrides
        if isinstance(entry, str) and canonicalize_name(Requirement(entry).name) == "numpy"
    ]
    assert not unscoped, (
        f"these numpy overrides apply to every package: {unscoped}. "
        'Scope each one, for example: '
        '{ package = { name = "pyctcdecode" }, dependencies = ["numpy>=2.3"] }'
    )


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


def test_numba():
    """librosa loads numba lazily, so test_librosa does not reach this import.

    numba refuses to import when numpy is newer than the cap numba declares.
    """
    import numba  # noqa: F401


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


def test_kenlm():
    import kenlm  # noqa: F401


def test_pyctcdecode():
    from pyctcdecode import build_ctcdecoder  # noqa: F401
