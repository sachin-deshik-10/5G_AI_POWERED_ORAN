# 🔌 **API Reference Guide**

[![API Status](https://img.shields.io/badge/API-v2.0-green.svg)](https://api.5g-oran-optimizer.ai)
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0-blue.svg)](docs/api/openapi.yaml)
[![Response Time](https://img.shields.io/badge/Response%20Time-<100ms-brightgreen.svg)]()

> **🎯 Complete API reference for the AI-Powered 5G Open RAN Optimizer**

## 📋 **Quick Navigation**

- [🚀 Getting Started](#-getting-started)
- [🔐 Authentication](#-authentication)
- [🧠 Cognitive Intelligence APIs](#-cognitive-intelligence-apis)
- [🔥 Edge AI APIs](#-edge-ai-apis)
- [🛡️ Security AI APIs](#️-security-ai-apis)
- [🤖 Autonomous Operations APIs](#-autonomous-operations-apis)
- [📊 Monitoring & Analytics APIs](#-monitoring--analytics-apis)
- [🔍 Error Handling](#-error-handling)
- [📡 WebSocket Events](#-websocket-events)
- [🔧 Rate Limiting](#-rate-limiting)

---

## 🚀 **Getting Started**

### **Base URL**

```
Production:  https://api.5g-oran-optimizer.ai/v2
Staging:     https://staging-api.5g-oran-optimizer.ai/v2
Local:       http://localhost:8000/v2
```

### **API Versions**

| Version | Status | Support Until | Features |
|---------|--------|---------------|----------|
| **v2.0** | ✅ Current | 2026-01-01 | Full feature set |
| **v1.0** | ⚠️ Deprecated | 2025-06-01 | Legacy support only |

### **Content Types**

```http
Content-Type: application/json
Accept: application/json
```

### **Quick Test**

```bash
curl -X GET "https://api.5g-oran-optimizer.ai/v2/health" \
  -H "Accept: application/json"
```

**Response:**

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2025-01-03T10:30:00Z",
  "services": {
    "cognitive_engine": "operational",
    "edge_ai": "operational",
    "security_ai": "operational",
    "autonomous_ops": "operational"
  }
}
```

---

## 🔐 **Authentication**

### **API Key Authentication**

```http
GET /v2/cognitive/optimize
Authorization: Bearer your-api-key-here
```

### **OAuth 2.0 (Enterprise)**

```bash
# Get access token
curl -X POST "https://api.5g-oran-optimizer.ai/oauth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=your-client-id&client_secret=your-client-secret"
```

### **Azure AD Integration**

```bash
# Azure AD token
curl -X GET "https://api.5g-oran-optimizer.ai/v2/cognitive/optimize" \
  -H "Authorization: Bearer azure-ad-token"
```

---

## 🧠 **Cognitive Intelligence APIs**

### **🎯 Network Optimization**

#### **POST** `/v2/cognitive/optimize`

Optimize network configuration using quantum-enhanced algorithms.

**Request:**

```json
{
  "network_id": "network-001",
  "optimization_objectives": ["throughput", "latency", "energy"],
  "constraints": {
    "max_power_w": 500,
    "min_reliability": 99.9,
    "budget_limit": 100000
  },
  "network_metrics": {
    "cells": [
      {
        "cell_id": "cell-001",
        "dl_throughput_mbps": 150.5,
        "ul_throughput_mbps": 75.2,
        "latency_ms": 12.1,
        "energy_consumption_w": 320.0,
        "load_percentage": 68.5
      }
    ]
  },
  "optimization_config": {
    "algorithm": "quantum_vqe",
    "confidence_threshold": 0.85,
    "max_iterations": 1000
  }
}
```

**Response:**

```json
{
  "optimization_id": "opt-12345",
  "status": "completed",
  "confidence_score": 0.94,
  "execution_time_ms": 1234,
  "results": {
    "recommended_actions": [
      {
        "cell_id": "cell-001",
        "action_type": "power_adjustment",
        "current_value": 320.0,
        "recommended_value": 280.0,
        "expected_improvement": {
          "energy_savings_percent": 12.5,
          "throughput_impact_percent": -2.1
        }
      }
    ],
    "performance_prediction": {
      "throughput_improvement_percent": 15.3,
      "latency_reduction_ms": 2.1,
      "energy_savings_percent": 18.7
    }
  }
}
```

#### **GET** `/v2/cognitive/optimization/{optimization_id}`

Get optimization results by ID.

**Response:**

```json
{
  "optimization_id": "opt-12345",
  "status": "completed",
  "created_at": "2025-01-03T10:30:00Z",
  "completed_at": "2025-01-03T10:31:15Z",
  "results": { /* ... */ }
}
```

### **🔮 Digital Twin**

#### **POST** `/v2/cognitive/digital-twin/create`

Create or update digital twin model.

**Request:**

```json
{
  "network_id": "network-001",
  "twin_config": {
    "fidelity_target": 0.95,
    "update_frequency_seconds": 30,
    "simulation_depth": "detailed"
  },
  "physical_data": {
    "topology": { /* network topology */ },
    "real_time_metrics": { /* current metrics */ }
  }
}
```

#### **GET** `/v2/cognitive/digital-twin/{twin_id}/predict`

Get predictions from digital twin.

**Query Parameters:**

- `prediction_horizon`: Time horizon for predictions (e.g., "1h", "24h")
- `scenario`: Scenario to simulate (e.g., "peak_traffic", "maintenance")

---

## 🔥 **Edge AI APIs**

### **⚡ Real-time Inference**

#### **POST** `/v2/edge/infer`

Perform ultra-low latency AI inference at the edge.

**Request:**

```json
{
  "edge_node_id": "edge-001",
  "inference_request": {
    "model_name": "network_anomaly_detector",
    "input_data": {
      "timestamp": "2025-01-03T10:30:00Z",
      "metrics": {
        "cpu_usage": 75.2,
        "memory_usage": 68.1,
        "network_throughput": 1250.5,
        "error_rate": 0.002
      }
    },
    "inference_config": {
      "precision": "fp16",
      "batch_size": 1,
      "max_latency_ms": 2
    }
  }
}
```

**Response:**

```json
{
  "inference_id": "inf-67890",
  "edge_node_id": "edge-001",
  "latency_ms": 1.2,
  "results": {
    "anomaly_score": 0.15,
    "classification": "normal",
    "confidence": 0.94,
    "feature_importance": {
      "cpu_usage": 0.3,
      "memory_usage": 0.2,
      "network_throughput": 0.4,
      "error_rate": 0.1
    }
  },
  "model_info": {
    "model_version": "v2.1.0",
    "model_size_mb": 25.6,
    "optimization": "onnx_quantized"
  }
}
```

### **🤝 Federated Learning**

#### **POST** `/v2/edge/federated/start-round`

Start a new federated learning round.

**Request:**

```json
{
  "federation_id": "fed-001",
  "round_config": {
    "target_participants": 10,
    "min_participants": 5,
    "max_duration_minutes": 30,
    "privacy_config": {
      "differential_privacy": true,
      "epsilon": 0.1,
      "delta": 1e-5
    }
  },
  "model_config": {
    "model_name": "traffic_predictor",
    "aggregation_method": "fedavg",
    "learning_rate": 0.001
  }
}
```

#### **POST** `/v2/edge/federated/submit-update`

Submit local model update for federated aggregation.

---

## 🛡️ **Security AI APIs**

### **🚨 Threat Detection**

#### **POST** `/v2/security/analyze`

Analyze network traffic for security threats.

**Request:**

```json
{
  "analysis_request": {
    "network_traffic": {
      "source_ip": "192.168.1.100",
      "destination_ip": "10.0.0.50",
      "protocol": "TCP",
      "port": 443,
      "packet_size": 1500,
      "flags": ["SYN", "ACK"],
      "payload_hash": "sha256:abc123..."
    },
    "context": {
      "time_window": "5m",
      "baseline_behavior": true,
      "threat_intelligence": true
    }
  }
}
```

**Response:**

```json
{
  "analysis_id": "sec-11111",
  "threat_level": "low",
  "risk_score": 0.15,
  "detection_time_ms": 45,
  "findings": [
    {
      "category": "behavioral_analysis",
      "severity": "info",
      "description": "Traffic pattern within normal baseline",
      "confidence": 0.89
    }
  ],
  "recommended_actions": [
    {
      "action": "continue_monitoring",
      "priority": "low",
      "automated": true
    }
  ]
}
```

### **🔐 Zero-Trust Verification**

#### **POST** `/v2/security/verify-access`

Perform zero-trust access verification.

**Request:**

```json
{
  "access_request": {
    "user_id": "user-001",
    "device_id": "device-001",
    "resource": "network-config",
    "action": "read",
    "context": {
      "location": "office_network",
      "time": "2025-01-03T10:30:00Z",
      "risk_factors": ["new_device", "unusual_time"]
    }
  }
}
```

---

## 🤖 **Autonomous Operations APIs**

### **🔧 Self-Healing**

#### **POST** `/v2/autonomous/heal`

Trigger autonomous healing process.

**Request:**

```json
{
  "healing_request": {
    "incident_id": "inc-001",
    "affected_components": ["cell-001", "cell-002"],
    "symptoms": [
      {
        "component": "cell-001",
        "metric": "throughput",
        "deviation": -25.5,
        "severity": "high"
      }
    ],
    "healing_config": {
      "max_actions": 5,
      "rollback_enabled": true,
      "approval_required": false
    }
  }
}
```

**Response:**

```json
{
  "healing_session_id": "heal-001",
  "status": "in_progress",
  "estimated_duration_minutes": 3,
  "planned_actions": [
    {
      "action_id": "action-001",
      "type": "parameter_adjustment",
      "target": "cell-001",
      "description": "Adjust antenna tilt to optimize coverage",
      "risk_level": "low",
      "rollback_plan": "revert_to_previous_value"
    }
  ]
}
```

### **📊 Performance Monitoring**

#### **GET** `/v2/autonomous/performance`

Get autonomous operations performance metrics.

**Query Parameters:**

- `time_range`: Time range for metrics (e.g., "1h", "24h", "7d")
- `component`: Specific component to monitor
- `metric_type`: Type of metrics ("healing", "optimization", "prediction")

---

## 📊 **Monitoring & Analytics APIs**

### **📈 System Metrics**

#### **GET** `/v2/monitoring/metrics`

Get system performance metrics.

**Query Parameters:**

- `start_time`: Start time (ISO 8601)
- `end_time`: End time (ISO 8601)
- `metrics`: Comma-separated list of metric names
- `aggregation`: Aggregation method ("avg", "sum", "max", "min")

**Response:**

```json
{
  "time_range": {
    "start": "2025-01-03T09:30:00Z",
    "end": "2025-01-03T10:30:00Z"
  },
  "metrics": {
    "api_latency_ms": {
      "avg": 85.2,
      "p95": 150.1,
      "p99": 280.5
    },
    "throughput_requests_per_second": {
      "avg": 1250.5,
      "max": 2100.0,
      "min": 800.2
    },
    "error_rate_percent": {
      "avg": 0.05,
      "max": 0.12
    }
  }
}
```

### **🎯 Analytics Dashboard Data**

#### **GET** `/v2/analytics/dashboard`

Get data for analytics dashboard.

**Response:**

```json
{
  "summary": {
    "total_optimizations": 15420,
    "avg_improvement_percent": 23.5,
    "active_edge_nodes": 156,
    "threat_incidents_blocked": 89
  },
  "recent_activity": [
    {
      "timestamp": "2025-01-03T10:25:00Z",
      "event_type": "optimization_completed",
      "description": "Network optimization improved throughput by 18.2%"
    }
  ]
}
```

---

## 🔍 **Error Handling**

### **Error Response Format**

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request contains invalid parameters",
    "details": {
      "field": "optimization_objectives",
      "issue": "Must contain at least one objective"
    },
    "trace_id": "trace-12345",
    "timestamp": "2025-01-03T10:30:00Z"
  }
}
```

### **HTTP Status Codes**

| Code | Description | Common Causes |
|------|-------------|---------------|
| **200** | Success | Request completed successfully |
| **400** | Bad Request | Invalid request parameters |
| **401** | Unauthorized | Missing or invalid authentication |
| **403** | Forbidden | Insufficient permissions |
| **404** | Not Found | Resource not found |
| **429** | Too Many Requests | Rate limit exceeded |
| **500** | Internal Server Error | Server-side error |
| **503** | Service Unavailable | Service temporarily unavailable |

### **Error Codes**

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `INVALID_REQUEST` | Request validation failed | Check request parameters |
| `AUTHENTICATION_FAILED` | Authentication error | Verify API key or token |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Reduce request rate |
| `RESOURCE_NOT_FOUND` | Requested resource not found | Check resource ID |
| `SERVICE_UNAVAILABLE` | Service temporarily down | Retry after delay |
| `QUOTA_EXCEEDED` | API quota exceeded | Upgrade plan or wait |

---

## 📡 **WebSocket Events**

### **Connection**

```javascript
const ws = new WebSocket('wss://api.5g-oran-optimizer.ai/v2/ws');
ws.send(JSON.stringify({
  type: 'authenticate',
  token: 'your-api-key'
}));
```

### **Real-time Events**

#### **Optimization Updates**

```json
{
  "event": "optimization_progress",
  "data": {
    "optimization_id": "opt-12345",
    "progress_percent": 75,
    "current_step": "evaluating_solutions",
    "estimated_completion": "2025-01-03T10:32:00Z"
  }
}
```

#### **Security Alerts**

```json
{
  "event": "security_alert",
  "data": {
    "alert_id": "alert-67890",
    "severity": "high",
    "threat_type": "anomaly_detected",
    "affected_components": ["cell-001"],
    "recommended_action": "isolate_component"
  }
}
```

#### **System Health Updates**

```json
{
  "event": "health_update",
  "data": {
    "component": "edge_ai",
    "status": "degraded",
    "metrics": {
      "response_time_ms": 150,
      "error_rate_percent": 2.1
    }
  }
}
```

---

## 🔧 **Rate Limiting**

### **Rate Limits**

| Plan | Requests/Minute | Burst Limit | WebSocket Connections |
|------|----------------|-------------|----------------------|
| **Free** | 100 | 200 | 5 |
| **Pro** | 1,000 | 2,000 | 50 |
| **Enterprise** | 10,000 | 20,000 | 500 |

### **Rate Limit Headers**

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641196800
X-RateLimit-Burst: 2000
```

### **Rate Limit Exceeded Response**

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Limit: 1000 requests per minute",
    "retry_after_seconds": 60
  }
}
```

---

## 📚 **SDKs & Libraries**

### **Python SDK**

```bash
pip install oran-optimizer-sdk
```

```python
from oran_optimizer import OptimizationClient

client = OptimizationClient(api_key='your-api-key')
result = client.optimize_network(network_data)
```

### **JavaScript SDK**

```bash
npm install @oran-optimizer/sdk
```

```javascript
import { OptimizationClient } from '@oran-optimizer/sdk';

const client = new OptimizationClient('your-api-key');
const result = await client.optimizeNetwork(networkData);
```

### **Go SDK**

```bash
go get github.com/oran-optimizer/go-sdk
```

```go
import "github.com/oran-optimizer/go-sdk"

client := sdk.NewClient("your-api-key")
result, err := client.OptimizeNetwork(networkData)
```

---

## 🔗 **Additional Resources**

- **📖 [OpenAPI Specification](openapi.yaml)**: Complete API specification
- **🎯 [Postman Collection](postman-collection.json)**: Ready-to-use API collection
- **📚 [Code Examples](../examples/)**: Implementation examples
- **💬 [Discord Support](https://discord.gg/5g-oran)**: Real-time API support
- **📧 [API Support](mailto:api-support@5g-oran-optimizer.ai)**: Dedicated API support team

---

*API Reference last updated: January 2025 | Version 2.0*

[![Test API](https://img.shields.io/badge/Test%20API-Postman-orange.svg)](postman-collection.json)
[![Get Support](https://img.shields.io/badge/Get%20Support-Discord-7289da.svg)](https://discord.gg/5g-oran)
