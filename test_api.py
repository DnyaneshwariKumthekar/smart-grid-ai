"""
Smart Grid AI - API Testing Suite
Quick tests to verify API functionality

Run with: python test_api.py
"""

import requests
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# ========================
# Configuration
# ========================

API_BASE_URL = "http://localhost:8000"
TIMEOUT = 10  # seconds

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

# ========================
# Test Data Generators
# ========================

def generate_features(count: int = 1, seed: int = 42) -> List[List[float]]:
    """Generate random feature vectors (31 features each)"""
    np.random.seed(seed)
    return np.random.randn(count, 31).tolist()


def get_sample_features() -> List[float]:
    """Get a single sample feature vector"""
    return generate_features(1)[0]


# ========================
# Test Functions
# ========================

def test_health() -> bool:
    """Test health check endpoint"""
    print(f"\n{BOLD}Testing Health Check...{RESET}")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"{GREEN}✓ Health check passed{RESET}")
            print(f"  Status: {data['status']}")
            print(f"  Uptime: {data['uptime_seconds']:.2f}s")
            print(f"  Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"{RED}✗ Health check failed: {response.status_code}{RESET}")
            return False
            
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_single_prediction() -> bool:
    """Test single prediction endpoint"""
    print(f"\n{BOLD}Testing Single Prediction...{RESET}")
    
    try:
        features = get_sample_features()
        
        response = requests.post(
            f"{API_BASE_URL}/predict/single",
            json={"features": features},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"{GREEN}✓ Single prediction successful{RESET}")
            print(f"  Prediction: {data['prediction']:.2f} kW")
            print(f"  Model: {data['model']}")
            print(f"  Confidence: {data['confidence']:.1%}")
            return True
        else:
            print(f"{RED}✗ Single prediction failed: {response.status_code}{RESET}")
            print(f"  Response: {response.json()}")
            return False
            
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_batch_prediction() -> bool:
    """Test batch prediction endpoint"""
    print(f"\n{BOLD}Testing Batch Prediction...{RESET}")
    
    try:
        # Test different batch sizes
        batch_sizes = [10, 100, 500]
        
        for batch_size in batch_sizes:
            features = generate_features(batch_size)
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict/batch",
                json={"features": features, "model": "moe"},
                timeout=TIMEOUT
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"{GREEN}✓ Batch {batch_size} samples: {data['processing_time_ms']:.2f}ms{RESET}")
                print(f"  Mean prediction: {data['mean_prediction']:.2f} kW")
                print(f"  Std prediction: {data['std_prediction']:.2f} kW")
                print(f"  Actual elapsed: {elapsed*1000:.2f}ms")
            else:
                print(f"{RED}✗ Batch {batch_size} failed: {response.status_code}{RESET}")
                return False
        
        return True
        
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_anomaly_detection() -> bool:
    """Test anomaly detection endpoint"""
    print(f"\n{BOLD}Testing Anomaly Detection...{RESET}")
    
    try:
        # Generate normal and anomalous data
        normal_features = generate_features(50, seed=42)
        anomalous_features = (np.random.randn(10, 31) + 10).tolist()  # Shifted
        all_features = normal_features + anomalous_features
        
        response = requests.post(
            f"{API_BASE_URL}/anomaly-detect",
            json={
                "features": all_features,
                "voting_threshold": 2
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"{GREEN}✓ Anomaly detection successful{RESET}")
            print(f"  Total samples: {data['total_samples']}")
            print(f"  Anomalies detected: {data['anomalies_detected']}")
            print(f"  Percentage anomaly: {data['percentage_anomaly']:.2f}%")
            print(f"  Anomaly indices: {data['anomaly_indices'][:5]}...")
            return True
        else:
            print(f"{RED}✗ Anomaly detection failed: {response.status_code}{RESET}")
            print(f"  Response: {response.json()}")
            return False
            
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_list_models() -> bool:
    """Test list models endpoint"""
    print(f"\n{BOLD}Testing List Models...{RESET}")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/models",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"{GREEN}✓ List models successful{RESET}")
            print(f"  Available models: {data['available_models']}")
            print(f"  Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"{RED}✗ List models failed: {response.status_code}{RESET}")
            return False
            
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_model_info() -> bool:
    """Test model info endpoint"""
    print(f"\n{BOLD}Testing Model Info...{RESET}")
    
    try:
        models = ["moe", "baseline", "anomaly"]
        
        for model_id in models:
            response = requests.get(
                f"{API_BASE_URL}/models/{model_id}",
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"{GREEN}✓ Model info for '{model_id}' loaded{RESET}")
                print(f"  Name: {data['name']}")
                print(f"  Type: {data['type']}")
            else:
                print(f"{RED}✗ Model info for '{model_id}' failed{RESET}")
                return False
        
        return True
        
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_error_handling() -> bool:
    """Test error handling"""
    print(f"\n{BOLD}Testing Error Handling...{RESET}")
    
    try:
        # Test 1: Invalid feature count
        print("  Testing invalid feature count...")
        response = requests.post(
            f"{API_BASE_URL}/predict/single",
            json={"features": [1.0] * 30},  # Should be 31
            timeout=TIMEOUT
        )
        if response.status_code == 422:
            print(f"  {GREEN}✓ Correctly rejected invalid input{RESET}")
        else:
            print(f"  {RED}✗ Should have rejected invalid input{RESET}")
            return False
        
        # Test 2: Non-existent model
        print("  Testing non-existent model...")
        response = requests.get(
            f"{API_BASE_URL}/models/nonexistent",
            timeout=TIMEOUT
        )
        if response.status_code == 404:
            print(f"  {GREEN}✓ Correctly returned 404 for missing model{RESET}")
        else:
            print(f"  {RED}✗ Should have returned 404{RESET}")
            return False
        
        return True
        
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


def test_performance() -> bool:
    """Performance benchmark test"""
    print(f"\n{BOLD}Testing Performance...{RESET}")
    
    try:
        # Warm-up request
        requests.post(
            f"{API_BASE_URL}/predict/single",
            json={"features": get_sample_features()},
            timeout=TIMEOUT
        )
        
        # Test single prediction latency
        print("  Benchmarking single prediction (100 requests)...")
        times = []
        for _ in range(100):
            start = time.time()
            requests.post(
                f"{API_BASE_URL}/predict/single",
                json={"features": get_sample_features()},
                timeout=TIMEOUT
            )
            times.append((time.time() - start) * 1000)
        
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)
        print(f"  {GREEN}✓ P50 latency: {p50:.2f}ms, P99 latency: {p99:.2f}ms{RESET}")
        
        # Test batch processing throughput
        print("  Benchmarking batch prediction (1000 samples)...")
        start = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"features": generate_features(1000)},
            timeout=30
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            throughput = 1000 / elapsed
            print(f"  {GREEN}✓ Throughput: {throughput:.0f} samples/sec{RESET}")
        
        return True
        
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")
        return False


# ========================
# Main Test Suite
# ========================

def run_all_tests() -> Dict[str, bool]:
    """Run all tests and return results"""
    
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}Smart Grid AI - API Test Suite{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")
    
    results = {
        "Health Check": test_health(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction(),
        "Anomaly Detection": test_anomaly_detection(),
        "List Models": test_list_models(),
        "Model Info": test_model_info(),
        "Error Handling": test_error_handling(),
        "Performance": test_performance(),
    }
    
    # Print summary
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}Test Summary{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{test_name:30s} {status}")
    
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"{GREEN}{BOLD}✓ All tests passed!{RESET}")
    else:
        print(f"{RED}{BOLD}✗ {total - passed} test(s) failed{RESET}")
    
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
    
    return results


# ========================
# CLI Interface
# ========================

if __name__ == "__main__":
    import sys
    
    try:
        # Check if server is running
        print("Checking if API server is running...")
        requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"{GREEN}✓ API server is running at {API_BASE_URL}{RESET}")
        
    except requests.exceptions.ConnectionError:
        print(f"{RED}✗ Cannot connect to API server at {API_BASE_URL}{RESET}")
        print("Start the server with: python inference_api.py")
        sys.exit(1)
    
    except requests.exceptions.Timeout:
        print(f"{RED}✗ Server connection timed out{RESET}")
        sys.exit(1)
    
    # Run tests
    results = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
