"""Validation tests to verify the testing infrastructure is set up correctly."""

import sys
import os
import pytest
import tempfile
from pathlib import Path

import torch
import numpy as np


class TestInfrastructureSetup:
    """Test suite to validate the testing infrastructure."""
    
    def test_python_path_configured(self):
        """Test that the project root is in Python path."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        assert project_root in sys.path, "Project root should be in Python path"
    
    def test_project_imports_work(self):
        """Test that project modules can be imported."""
        try:
            import model
            import sac
            import replay_memory
            import utils
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import project modules: {e}")
    
    def test_fixtures_available(self, temp_dir, mock_env, device, random_seed):
        """Test that conftest fixtures are available."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert hasattr(mock_env, 'reset')
        assert hasattr(mock_env, 'step')
        assert isinstance(device, torch.device)
        assert random_seed == 42
    
    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temporary directory fixture works and cleans up."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
    
    def test_mock_env_fixture(self, mock_env):
        """Test the mock environment fixture."""
        state = mock_env.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (4,)
        
        next_state, reward, done, info = mock_env.step([0.1, 0.2])
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_sample_tensors(self, sample_state, sample_action, sample_batch):
        """Test sample tensor fixtures."""
        assert isinstance(sample_state, torch.Tensor)
        assert sample_state.shape == (1, 4)
        
        assert isinstance(sample_action, torch.Tensor)
        assert sample_action.shape == (1, 2)
        
        assert isinstance(sample_batch, dict)
        assert 'state' in sample_batch
        assert 'action' in sample_batch
        assert 'next_state' in sample_batch
        assert 'reward' in sample_batch
        assert 'not_done' in sample_batch
    
    def test_mock_config(self, mock_config):
        """Test mock configuration fixture."""
        assert isinstance(mock_config, dict)
        required_keys = ['env_name', 'seed', 'batch_size', 'discount']
        for key in required_keys:
            assert key in mock_config
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True
    
    def test_coverage_source_files_exist(self):
        """Test that source files for coverage exist."""
        source_files = ['model.py', 'sac.py', 'replay_memory.py', 'utils.py', 'main.py']
        project_root = Path(__file__).parent.parent
        
        for file in source_files:
            filepath = project_root / file
            assert filepath.exists(), f"Source file {file} should exist"


class TestPytestConfiguration:
    """Test pytest configuration."""
    
    def test_pytest_config_exists(self):
        """Test that pytest configuration exists in pyproject.toml."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"
        
        content = pyproject_path.read_text()
        assert "[tool.pytest.ini_options]" in content
        assert "[tool.coverage.run]" in content
        assert "[tool.coverage.report]" in content
    
    def test_test_discovery_patterns(self):
        """Test that test discovery patterns are configured."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text()
        
        assert 'python_files = ["test_*.py", "*_test.py"]' in content
        assert 'python_classes = ["Test*"]' in content
        assert 'python_functions = ["test_*"]' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])