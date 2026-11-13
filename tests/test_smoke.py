import earworm


def test_package_imports():
    assert earworm.__version__ == "0.1.0"


def test_core_dependencies_import():
    import librosa  # noqa: F401
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import soundfile  # noqa: F401
