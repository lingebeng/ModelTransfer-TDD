import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("JAX_LOG_LEVEL", "ERROR")
