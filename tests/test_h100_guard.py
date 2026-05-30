from stageml.h100_guard import current_h100_environment


def test_current_h100_environment_has_expected_keys():
    env = current_h100_environment().to_dict()
    for key in ["cuda_available", "device_name", "capability", "torch_version", "torch_cuda_version", "total_memory_gb"]:
        assert key in env
