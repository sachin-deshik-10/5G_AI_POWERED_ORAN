"""
Performance and Load Testing Suite for 5G O-RAN Optimizer
Comprehensive testing for API performance, system scalability, and stress testing
"""

import pytest
import asyncio
import aiohttp
import time
import statistics
import concurrent.futures
from typing import List, Dict, Any
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PerformanceTestSuite:
    """Performance testing suite for the 5G O-RAN Optimizer"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    async def test_api_response_time(self, endpoint: str, payload: Dict = None, 
                                   num_requests: int = 100) -> Dict[str, float]:
        """Test API response time under normal load"""
        response_times = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for _ in range(num_requests):
                if payload:
                    task = self._make_post_request(session, endpoint, payload)
                else:
                    task = self._make_get_request(session, endpoint)
                tasks.append(task)
            
            # Execute requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate response times
            for response in responses:
                if isinstance(response, dict) and 'response_time' in response:
                    response_times.append(response['response_time'])
        
        return {
            'avg_response_time': statistics.mean(response_times),
            'median_response_time': statistics.median(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'total_requests': len(response_times),
            'success_rate': len(response_times) / num_requests
        }
    
    async def _make_get_request(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        """Make GET request and measure response time"""
        start_time = time.time()
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                await response.text()
                return {
                    'response_time': (time.time() - start_time) * 1000,  # ms
                    'status_code': response.status,
                    'success': response.status == 200
                }
        except Exception as e:
            return {
                'response_time': (time.time() - start_time) * 1000,
                'error': str(e),
                'success': False
            }
    
    async def _make_post_request(self, session: aiohttp.ClientSession, 
                               endpoint: str, payload: Dict) -> Dict:
        """Make POST request and measure response time"""
        start_time = time.time()
        try:
            async with session.post(
                f"{self.base_url}{endpoint}", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                await response.text()
                return {
                    'response_time': (time.time() - start_time) * 1000,  # ms
                    'status_code': response.status,
                    'success': response.status == 200
                }
        except Exception as e:
            return {
                'response_time': (time.time() - start_time) * 1000,
                'error': str(e),
                'success': False
            }
    
    async def stress_test_optimization_endpoint(self, 
                                              concurrent_users: List[int] = [10, 50, 100, 200],
                                              duration_seconds: int = 60) -> Dict:
        """Stress test the optimization endpoint"""
        optimization_payload = {
            "network_metrics": {
                "dl_throughput_mbps": 85.5,
                "ul_throughput_mbps": 45.2,
                "latency_ms": 4.8,
                "packet_loss_percent": 0.01,
                "energy_consumption_w": 75.3,
                "cpu_utilization": 65.0,
                "memory_utilization": 72.5,
                "user_count": 250,
                "spectrum_efficiency": 5.2,
                "beamforming_gain": 12.5,
                "mimo_rank": 4
            },
            "slice_config": {
                "slice_type": "eMBB",
                "priority": "high",
                "bandwidth_requirement_mbps": 100,
                "latency_requirement_ms": 10,
                "reliability_requirement": 99.9
            }
        }
        
        stress_results = {}
        
        for num_users in concurrent_users:
            print(f"Testing with {num_users} concurrent users...")
            
            # Run stress test for specified duration
            end_time = time.time() + duration_seconds
            total_requests = 0
            successful_requests = 0
            response_times = []
            
            async with aiohttp.ClientSession() as session:
                while time.time() < end_time:
                    # Create concurrent requests
                    tasks = []
                    for _ in range(num_users):
                        task = self._make_post_request(
                            session, "/api/optimize", optimization_payload
                        )
                        tasks.append(task)
                    
                    # Execute batch of requests
                    batch_start = time.time()
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    batch_duration = time.time() - batch_start
                    
                    # Process results
                    for response in responses:
                        total_requests += 1
                        if isinstance(response, dict):
                            if response.get('success', False):
                                successful_requests += 1
                                response_times.append(response['response_time'])
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)
            
            stress_results[num_users] = {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
                'requests_per_second': total_requests / duration_seconds,
                'successful_rps': successful_requests / duration_seconds
            }
        
        return stress_results
    
    async def test_memory_and_cpu_usage(self) -> Dict:
        """Test system resource usage during load"""
        import psutil
        
        # Baseline measurements
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Run load test
        load_payload = {
            "network_metrics": {
                "dl_throughput_mbps": 100.0,
                "ul_throughput_mbps": 50.0,
                "latency_ms": 5.0,
                "packet_loss_percent": 0.02,
                "energy_consumption_w": 80.0,
                "cpu_utilization": 70.0,
                "memory_utilization": 75.0,
                "user_count": 300,
                "spectrum_efficiency": 6.0,
                "beamforming_gain": 15.0,
                "mimo_rank": 6
            }
        }
        
        # Monitor during load
        memory_usage = []
        cpu_usage = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(50):  # 50 concurrent requests
                task = self._make_post_request(session, "/api/optimize", load_payload)
                tasks.append(task)
            
            # Monitor resources while requests are processing
            start_monitoring = time.time()
            while time.time() - start_monitoring < 10:  # Monitor for 10 seconds
                memory_usage.append(psutil.virtual_memory().percent)
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
                await asyncio.sleep(0.5)
            
            # Complete requests
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'initial_memory_percent': initial_memory,
            'initial_cpu_percent': initial_cpu,
            'peak_memory_percent': max(memory_usage),
            'avg_memory_percent': statistics.mean(memory_usage),
            'peak_cpu_percent': max(cpu_usage),
            'avg_cpu_percent': statistics.mean(cpu_usage),
            'memory_increase': max(memory_usage) - initial_memory,
            'cpu_increase': max(cpu_usage) - initial_cpu
        }
    
    async def test_websocket_performance(self, num_connections: int = 10, 
                                       duration_seconds: int = 30) -> Dict:
        """Test WebSocket performance and connection handling"""
        import websockets
        
        connection_times = []
        message_counts = []
        disconnection_times = []
        
        async def websocket_client(client_id: int):
            """Individual WebSocket client"""
            messages_received = 0
            connection_start = time.time()
            
            try:
                uri = self.base_url.replace('http://', 'ws://') + '/ws/real-time'
                async with websockets.connect(uri) as websocket:
                    connection_time = (time.time() - connection_start) * 1000
                    connection_times.append(connection_time)
                    
                    # Listen for messages for specified duration
                    end_time = time.time() + duration_seconds
                    while time.time() < end_time:
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(), timeout=1.0
                            )
                            messages_received += 1
                        except asyncio.TimeoutError:
                            continue
                        except Exception:
                            break
                    
                    message_counts.append(messages_received)
                    
            except Exception as e:
                print(f"WebSocket client {client_id} error: {e}")
        
        # Run multiple WebSocket clients concurrently
        tasks = [websocket_client(i) for i in range(num_connections)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'num_connections': num_connections,
            'avg_connection_time_ms': statistics.mean(connection_times) if connection_times else 0,
            'max_connection_time_ms': max(connection_times) if connection_times else 0,
            'total_messages_received': sum(message_counts),
            'avg_messages_per_connection': statistics.mean(message_counts) if message_counts else 0,
            'messages_per_second': sum(message_counts) / duration_seconds if duration_seconds > 0 else 0
        }
    
    def generate_performance_report(self, results: Dict) -> str:
        """Generate a comprehensive performance report"""
        report = []
        report.append("=" * 60)
        report.append("5G O-RAN OPTIMIZER - PERFORMANCE TEST REPORT")
        report.append("=" * 60)
        report.append(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # API Performance Results
        if 'api_performance' in results:
            api_results = results['api_performance']
            report.append("API PERFORMANCE METRICS:")
            report.append("-" * 30)
            report.append(f"Average Response Time: {api_results['avg_response_time']:.2f} ms")
            report.append(f"95th Percentile: {api_results['p95_response_time']:.2f} ms")
            report.append(f"99th Percentile: {api_results['p99_response_time']:.2f} ms")
            report.append(f"Success Rate: {api_results['success_rate']:.1%}")
            report.append("")
        
        # Stress Test Results
        if 'stress_test' in results:
            stress_results = results['stress_test']
            report.append("STRESS TEST RESULTS:")
            report.append("-" * 30)
            for users, metrics in stress_results.items():
                report.append(f"{users} Concurrent Users:")
                report.append(f"  - Requests/sec: {metrics['requests_per_second']:.1f}")
                report.append(f"  - Success Rate: {metrics['success_rate']:.1%}")
                report.append(f"  - Avg Response: {metrics['avg_response_time']:.2f} ms")
                report.append("")
        
        # Resource Usage
        if 'resource_usage' in results:
            resource_results = results['resource_usage']
            report.append("RESOURCE USAGE:")
            report.append("-" * 30)
            report.append(f"Peak Memory Usage: {resource_results['peak_memory_percent']:.1f}%")
            report.append(f"Peak CPU Usage: {resource_results['peak_cpu_percent']:.1f}%")
            report.append(f"Memory Increase: {resource_results['memory_increase']:.1f}%")
            report.append(f"CPU Increase: {resource_results['cpu_increase']:.1f}%")
            report.append("")
        
        # WebSocket Performance
        if 'websocket_performance' in results:
            ws_results = results['websocket_performance']
            report.append("WEBSOCKET PERFORMANCE:")
            report.append("-" * 30)
            report.append(f"Connections Tested: {ws_results['num_connections']}")
            report.append(f"Avg Connection Time: {ws_results['avg_connection_time_ms']:.2f} ms")
            report.append(f"Messages/sec: {ws_results['messages_per_second']:.1f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

# Pytest integration
@pytest.mark.asyncio
async def test_api_performance():
    """Test API performance under normal load"""
    test_suite = PerformanceTestSuite()
    
    # Test health endpoint
    health_results = await test_suite.test_api_response_time("/health", num_requests=50)
    assert health_results['avg_response_time'] < 100  # Less than 100ms
    assert health_results['success_rate'] > 0.95  # 95% success rate
    
    # Test optimization endpoint
    optimization_payload = {
        "network_metrics": {
            "dl_throughput_mbps": 75.0,
            "ul_throughput_mbps": 35.0,
            "latency_ms": 6.0,
            "packet_loss_percent": 0.01,
            "energy_consumption_w": 70.0,
            "cpu_utilization": 60.0,
            "memory_utilization": 65.0,
            "user_count": 200,
            "spectrum_efficiency": 4.5,
            "beamforming_gain": 10.0,
            "mimo_rank": 3
        }
    }
    
    opt_results = await test_suite.test_api_response_time(
        "/api/optimize", optimization_payload, num_requests=25
    )
    assert opt_results['avg_response_time'] < 1000  # Less than 1 second
    assert opt_results['success_rate'] > 0.90  # 90% success rate

@pytest.mark.asyncio
async def test_stress_scenarios():
    """Test system under stress conditions"""
    test_suite = PerformanceTestSuite()
    
    # Light stress test
    stress_results = await test_suite.stress_test_optimization_endpoint(
        concurrent_users=[10, 25], duration_seconds=30
    )
    
    # Verify system handles concurrent load
    for users, metrics in stress_results.items():
        assert metrics['success_rate'] > 0.80  # 80% success rate under stress
        assert metrics['avg_response_time'] < 2000  # Less than 2 seconds

@pytest.mark.asyncio
async def test_resource_consumption():
    """Test system resource consumption"""
    test_suite = PerformanceTestSuite()
    
    resource_results = await test_suite.test_memory_and_cpu_usage()
    
    # Verify reasonable resource usage
    assert resource_results['peak_memory_percent'] < 85  # Less than 85% memory
    assert resource_results['peak_cpu_percent'] < 90  # Less than 90% CPU

if __name__ == "__main__":
    async def run_full_performance_suite():
        """Run complete performance test suite"""
        test_suite = PerformanceTestSuite()
        
        print("Starting comprehensive performance testing...")
        
        # Run all tests
        api_results = await test_suite.test_api_response_time("/health", num_requests=100)
        stress_results = await test_suite.stress_test_optimization_endpoint(
            concurrent_users=[10, 50, 100], duration_seconds=60
        )
        resource_results = await test_suite.test_memory_and_cpu_usage()
        ws_results = await test_suite.test_websocket_performance(
            num_connections=10, duration_seconds=30
        )
        
        # Compile results
        all_results = {
            'api_performance': api_results,
            'stress_test': stress_results,
            'resource_usage': resource_results,
            'websocket_performance': ws_results
        }
        
        # Generate and save report
        report = test_suite.generate_performance_report(all_results)
        print(report)
        
        # Save to file
        with open('performance_test_report.txt', 'w') as f:
            f.write(report)
        
        print("\nPerformance test completed. Report saved to 'performance_test_report.txt'")
    
    # Run the test suite
    asyncio.run(run_full_performance_suite())
