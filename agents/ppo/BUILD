py_library(
    name = "ppo",
    srcs = [
        "__init__.py",
        "algorithm.py",
        "memory.py",
        "normalize.py",
        "utility.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//learning/brain/public:tensorflow_gpu_deps",
        "//learning/brain/public:tensorflow_std_ops",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)
