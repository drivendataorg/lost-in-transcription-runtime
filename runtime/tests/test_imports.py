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


def _arpa(order):
    """Build a tiny ARPA model of the given order.

    Held inline so the tests need no data file. The doubled backslashes are
    the \\data\\ and \\end\\ section markers that the format requires.
    """
    lines = ["", "\\data\\", "ngram 1=4"]
    lines += [f"ngram {k}=1" for k in range(2, order + 1)]
    lines += ["", "\\1-grams:"]
    lines += ["-1.0\t<unk>\t0.0", "0.0\t<s>\t-0.5", "-1.0\t</s>", "-0.7\ta\t-0.3", ""]
    for k in range(2, order + 1):
        gram = " ".join(["a"] * k)
        backoff = "" if k == order else "\t-0.3"   # the top order carries none
        lines += [f"\\{k}-grams:", f"-0.4\t{gram}{backoff}", ""]
    lines += ["\\end\\", ""]
    return "\n".join(lines)


def test_kenlm(tmp_path):
    """Load an ARPA model plain and compressed to check what kenlm can read.

    kenlm's setup.py probes for zlib.h, bzlib.h and lzma.h with g++ at build
    time. It drops the matching format when a header is missing, and the build
    still succeeds. An import-only test passes either way. The break would then
    appear when a submission loads a compressed model, and the container has no
    internet access to correct it. Most ARPA files ship compressed because they
    are large, so read one of each form here.
    """
    import bz2
    import gzip
    import lzma

    import kenlm

    plain = tmp_path / "model.arpa"
    plain.write_text(_arpa(2))
    expected = kenlm.Model(str(plain)).score("a")

    for suffix, opener in ((".gz", gzip.open), (".bz2", bz2.open), (".xz", lzma.open)):
        packed = tmp_path / f"model.arpa{suffix}"
        with opener(packed, "wt") as handle:
            handle.write(_arpa(2))
        score = kenlm.Model(str(packed)).score("a")
        assert score == expected, f"kenlm read {suffix} but scored it differently"


def test_kenlm_max_order(tmp_path):
    """Check that kenlm compiled with the order the Dockerfile asks for.

    kenlm reads MAX_ORDER at build time and bakes it in, defaulting to 6. A
    model above that limit raises FormatLoadException and nothing can raise the
    limit afterward. Character-level n-gram LMs, the usual pairing with
    pyctcdecode, often run to order 7 or higher.

    Read the wanted order from the environment rather than hard-coding it. The
    Dockerfile sets MAX_ORDER, and uv can hand back a wheel it built before
    that value changed, so comparing the two is what catches the stale wheel.
    """
    import os

    import kenlm

    want = int(os.environ.get("MAX_ORDER", "10"))
    model = tmp_path / f"order{want}.arpa"
    model.write_text(_arpa(want))
    assert kenlm.Model(str(model)).order == want


def test_pyctcdecode():
    from pyctcdecode import build_ctcdecoder  # noqa: F401
