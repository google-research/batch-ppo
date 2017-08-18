py_binary(
    name = "train",
    srcs = [
        "configs.py",
        "networks.py",
        "train.py",
        "utility.py",
    ],
    deps = [
        "//robotics/reinforcement_learning/agents/ppo",
        "//robotics/reinforcement_learning/agents/tools",
        "//robotics/reinforcement_learning/environments/gym_mujoco",
        "//third_party/py/ruamel",
    ],
)

py_binary(
    name = "visualize",
    srcs = [
        "configs.py",
        "networks.py",
        "utility.py",
        "visualize.py",
    ],
    deps = [
        "//robotics/reinforcement_learning/agents/ppo",
        "//robotics/reinforcement_learning/agents/tools",
        "//robotics/reinforcement_learning/environments/gym_mujoco",
        "//third_party/py/ruamel",
    ],
)

py_test(
    name = "train_ppo_test",
    size = "large",
    srcs = [
        "train_ppo_test.py",
    ],
    tags = [
        "noasan",
        "notsan",
    ],
    deps = [
        ":train",
    ],
)
