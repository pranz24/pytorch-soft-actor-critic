import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import torch
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_env():
    """Create a mock Gym environment."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.observation_space.shape = (4,)
    env.action_space = MagicMock()
    env.action_space.shape = (2,)
    env.reset.return_value = np.zeros(4)
    env.step.return_value = (np.zeros(4), 0.0, False, {})
    return env


@pytest.fixture
def device():
    """Get the appropriate device (CPU for testing)."""
    return torch.device("cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def sample_state():
    """Generate a sample state tensor."""
    return torch.randn(1, 4)


@pytest.fixture
def sample_action():
    """Generate a sample action tensor."""
    return torch.randn(1, 2)


@pytest.fixture
def sample_batch():
    """Generate a sample batch of experiences."""
    batch_size = 32
    state_dim = 4
    action_dim = 2
    
    return {
        'state': torch.randn(batch_size, state_dim),
        'action': torch.randn(batch_size, action_dim),
        'next_state': torch.randn(batch_size, state_dim),
        'reward': torch.randn(batch_size, 1),
        'not_done': torch.ones(batch_size, 1)
    }


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = {
        'env_name': 'test_env',
        'seed': 42,
        'start_timesteps': 1000,
        'eval_freq': 5000,
        'max_timesteps': 100000,
        'expl_noise': 0.1,
        'batch_size': 256,
        'discount': 0.99,
        'tau': 0.005,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'policy_freq': 2,
        'save_model': True,
        'load_model': ""
    }
    return config


@pytest.fixture
def mock_tensorboard_writer(tmp_path):
    """Create a mock TensorBoard SummaryWriter."""
    writer = MagicMock()
    writer.add_scalar = MagicMock()
    writer.close = MagicMock()
    return writer


@pytest.fixture(autouse=True)
def cleanup_torch():
    """Clean up PyTorch resources after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    captured_output = io.StringIO()
    monkeypatch.setattr('sys.stdout', captured_output)
    return captured_output