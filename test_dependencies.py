#!/usr/bin/env python3
"""
Test script to verify compatibility with updated dependencies.
Tests basic imports and functionality of key modules.
"""

import sys
import traceback

def test_basic_imports():
    """Test that all basic dependencies can be imported."""
    print("Testing basic dependency imports...")
    
    dependencies = [
        "numpy",
        "torch", 
        "matplotlib",
        "scipy",
        "sklearn",
        "numba",
        "seaborn",
        "plotly",
        "altair",
        "tqdm",
        "lmfit",
        "hyperspy",
        "ipywidgets",
        "notebook",
        "jupyterlab",
    ]
    
    failed = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except Exception as e:
            print(f"  ✗ {dep}: {e}")
            failed.append(dep)
    
    if failed:
        print(f"\n⚠ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All basic dependencies imported successfully")
        return True

def test_numpy_version():
    """Test numpy version and basic functionality."""
    print("\nTesting NumPy...")
    try:
        import numpy as np
        print(f"  NumPy version: {np.__version__}")
        
        # Test basic numpy 2.x functionality
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.dtype == np.int64 or arr.dtype == np.int32
        
        # Test float32 (used in the code)
        arr_f32 = arr.astype(np.float32)
        assert arr_f32.dtype == np.float32
        
        print("  ✓ NumPy basic operations work")
        return True
    except Exception as e:
        print(f"  ✗ NumPy test failed: {e}")
        traceback.print_exc()
        return False

def test_torch_version():
    """Test PyTorch version and Softmax with dim parameter."""
    print("\nTesting PyTorch...")
    try:
        import torch
        import torch.nn as nn
        print(f"  PyTorch version: {torch.__version__}")
        
        # Test Softmax with dim parameter (our code change)
        softmax = nn.Softmax(dim=-1)
        test_tensor = torch.randn(2, 5)
        output = softmax(test_tensor)
        
        # Verify softmax sums to 1 along the specified dimension
        assert torch.allclose(output.sum(dim=-1), torch.ones(2))
        
        print("  ✓ PyTorch Softmax(dim=-1) works correctly")
        return True
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")
        traceback.print_exc()
        return False

def test_numba_numpy_compat():
    """Test Numba compatibility with NumPy."""
    print("\nTesting Numba + NumPy compatibility...")
    try:
        import numpy as np
        import numba
        from numba import jit
        
        print(f"  Numba version: {numba.__version__}")
        print(f"  NumPy version: {np.__version__}")
        
        # Simple JIT compiled function
        @jit(nopython=True)
        def add_arrays(a, b):
            return a + b
        
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])
        result = add_arrays(arr1, arr2)
        
        expected = np.array([5.0, 7.0, 9.0])
        assert np.allclose(result, expected)
        
        print("  ✓ Numba JIT compilation works with NumPy")
        return True
    except Exception as e:
        print(f"  ✗ Numba test failed: {e}")
        traceback.print_exc()
        return False

def test_sigma_imports():
    """Test that sigma package can be imported."""
    print("\nTesting SIGMA package imports...")
    try:
        import sigma
        print(f"  ✓ sigma")
        
        from sigma import models
        print(f"  ✓ sigma.models")
        
        from sigma import utils
        print(f"  ✓ sigma.utils")
        
        from sigma import gui
        print(f"  ✓ sigma.gui")
        
        print("\n✓ All SIGMA modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ SIGMA import failed: {e}")
        traceback.print_exc()
        return False

def test_autoencoder_model():
    """Test that AutoEncoder model can be instantiated with updated PyTorch."""
    print("\nTesting AutoEncoder model...")
    try:
        import torch
        from sigma.models.autoencoder import AutoEncoder, VariationalAutoEncoder
        
        # Test AutoEncoder
        model = AutoEncoder(in_channel=10, hidden_layer_sizes=(64, 32, 16))
        test_input = torch.randn(5, 10)
        output = model(test_input)
        
        assert output.shape == (5, 10), f"Expected shape (5, 10), got {output.shape}"
        
        # Verify softmax output sums to 1 (due to Softmax layer)
        assert torch.allclose(output.sum(dim=-1), torch.ones(5), atol=1e-5)
        
        print("  ✓ AutoEncoder works correctly")
        
        # Test VariationalAutoEncoder
        vae = VariationalAutoEncoder(in_channel=10, hidden_layer_sizes=(64, 32, 16))
        mu, logvar, z, x_recon = vae(test_input)
        
        assert x_recon.shape == (5, 10)
        assert mu.shape == (5, 2)
        assert logvar.shape == (5, 2)
        assert z.shape == (5, 2)
        
        print("  ✓ VariationalAutoEncoder works correctly")
        return True
    except Exception as e:
        print(f"  ✗ AutoEncoder test failed: {e}")
        traceback.print_exc()
        return False

def test_scikit_learn():
    """Test scikit-learn compatibility."""
    print("\nTesting scikit-learn...")
    try:
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
        from sklearn.cluster import KMeans, HDBSCAN
        from sklearn.decomposition import NMF
        import numpy as np
        
        # Test basic clustering
        X = np.random.randn(100, 5)
        
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        
        assert len(labels) == 100
        assert labels.min() >= 0
        assert labels.max() < 3
        
        print("  ✓ scikit-learn clustering works")
        return True
    except Exception as e:
        print(f"  ✗ scikit-learn test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("SIGMA Dependency Compatibility Test Suite")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("NumPy", test_numpy_version),
        ("PyTorch", test_torch_version),
        ("Numba + NumPy", test_numba_numpy_compat),
        ("SIGMA Package", test_sigma_imports),
        ("AutoEncoder Model", test_autoencoder_model),
        ("scikit-learn", test_scikit_learn),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            traceback.print_exc()
            results[name] = False
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
