py_library(
    name = "tools",
    srcs = [
        "__init__.py",
        "attr_dict.py",
        "batch_env.py",
        "count_weights.py",
        "in_graph_batch_env.py",
        "in_graph_env.py",
        "loop.py",
        "mock_algorithm.py",
        "mock_environment.py",
        "simulate.py",
        "streaming_mean.py",
        "wrappers.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/gym",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_test(
    name = "attr_dict_test",
    srcs = [
        "attr_dict_test.py",
    ],
    deps = [
        ":tools",
    ],
)

py_test(
    name = "count_weights_test",
    srcs = [
        "count_weights_test.py",
    ],
    deps = [
        ":tools",
    ],
)

py_test(
    name = "loop_test",
    srcs = [
        "loop_test.py",
    ],
    deps = [
        ":tools",
    ],
)

py_test(
    name = "simulate_test",
    srcs = [
        "simulate_test.py",
    ],
    deps = [
        ":tools",
    ],
)

py_test(
    name = "wrappers_test",
    srcs = [
        "wrappers_test.py",
    ],
    deps = [
        ":tools",
    ],
)
