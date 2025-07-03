# 🚀 AI-Powered 5G OpenRAN Optimizer - Advanced Setup Guide

## 📋 Prerequisites

### System Requirements

- **Python 3.9+** (3.11 recommended for best performance)
- **Docker & Docker Compose** (for containerized deployment)
- **Kubernetes** (for production deployment)
- **Redis** (for real-time caching)
- **PostgreSQL** (for production database)

### Hardware Requirements

- **Minimum:** 8GB RAM, 4 CPU cores, 50GB storage
- **Recommended:** 16GB RAM, 8 CPU cores, 100GB SSD storage
- **Production:** 32GB+ RAM, 16+ CPU cores, high-speed storage

## 🔧 Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/sachin-deshik-10/5G_AI_POWERED_ORAN.git
cd 5G_AI_POWERED_ORAN

# Run automated setup
python setup.py

# Start services
python demo.py
```

### Option 2: Manual Setup

```bash
# Create Python environment
python -m venv network_optimization_env
source network_optimization_env/bin/activate  # Linux/Mac
# or
network_optimization_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install additional AI/ML packages (optional but recommended)
pip install pytorch-lightning ray[default] optuna darts neuralprophet torch-geometric mlflow wandb

# Run the complete pipeline
python demo.py
```

### Option 3: Docker Setup (Production-Ready)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individual container
docker build -t 5g-oran-optimizer .
docker run -p 8000:8000 -p 8501:8501 5g-oran-optimizer
```

## 🎯 Usage Examples

### Basic Usage

```bash
# Run complete AI optimization pipeline
python demo.py

# Individual components
python src/preprocess.py --input "computing_datasets/datasets_unpin/realistic_computing.csv" --output "preprocessed_data.csv"
python src/train.py --input "preprocessed_data.csv"
python src/evaluate.py --input "preprocessed_data.csv" --models "models"
python src/optimize.py --input "computing_datasets/datasets_unpin/realistic_computing.csv" --output "optimized_results.csv"
```

### Advanced AI Features

```python
from src.models.advanced_ai_optimizer import AdvancedAIOptimizer, OptimizationConfig

# Configure advanced AI optimization
config = OptimizationConfig(
    use_transformer=True,
    use_reinforcement_learning=True,
    use_federated_learning=True,
    use_graph_neural_networks=True
)

# Initialize advanced optimizer
optimizer = AdvancedAIOptimizer(config)

# Run optimization
results = optimizer.optimize(network_data, method='transformer')
```

### Production API Server

```bash
# Start FastAPI server with advanced features
python -m uvicorn api.api_server:app --host 0.0.0.0 --port 8000 --workers 4

# Test API endpoints
curl -X POST "http://localhost:8000/api/v1/optimize" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -d '{"network_metrics": {...}, "slice_config": {...}}'
```

### Real-time Dashboard

```bash
# Launch advanced Streamlit dashboard
streamlit run dashboard/real_time_monitor.py --server.port 8501

# Access at: http://localhost:8501
```

## 🧪 Testing

### Run All Tests

```bash
# Unit tests
python -m pytest tests/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Performance tests
python -c "
import time
from src.main import main
start = time.time()
main('computing_datasets/datasets_unpin/realistic_computing.csv')
print(f'Execution time: {time.time() - start:.2f}s')
"
```

### Load Testing

```bash
# Install locust for load testing
pip install locust

# Run load tests (if you have a locust file)
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🔒 Security Setup

### API Authentication

```bash
# Generate JWT token for API access
python -c "
from api.api_server import create_access_token
token = create_access_token({'sub': 'admin', 'role': 'admin'})
print(f'JWT Token: {token}')
"
```

### Environment Variables

```bash
# Create .env file
cat > .env << EOF
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/oran_db
REDIS_URL=redis://localhost:6379/0
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF
```

## 🚀 Production Deployment

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/production.yaml

# Check deployment status
kubectl get pods -l app=5g-oran-optimizer

# Scale deployment
kubectl scale deployment 5g-oran-optimizer --replicas=5
```

### Docker Compose (Local Production)

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/oran_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: oran_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## 📊 Monitoring & Observability

### Prometheus Metrics

```bash
# Access metrics endpoint
curl http://localhost:8000/metrics

# Example metrics:
# - api_requests_total
# - api_request_duration_seconds
# - websocket_connections_active
# - optimization_requests_total
```

### Structured Logging

```python
# View structured logs
tail -f logs/application.log | jq '.'

# Example log entry:
{
  "timestamp": "2025-07-03T14:30:00Z",
  "level": "INFO",
  "message": "Optimization completed",
  "cell_id": "5G_Cell_07",
  "optimization_type": "throughput",
  "duration_ms": 45
}
```

## 🔧 Configuration

### Advanced Configuration

```yaml
# config.yml
ai_optimizer:
  transformer:
    hidden_dim: 512
    num_heads: 16
    num_layers: 8
  reinforcement_learning:
    algorithm: "PPO"
    learning_rate: 3e-4
  federated_learning:
    num_clients: 10
    global_rounds: 100

api_server:
  rate_limiting:
    requests_per_minute: 100
  security:
    jwt_expire_minutes: 30
  monitoring:
    metrics_enabled: true

dashboard:
  refresh_rate: 5
  enable_forecasting: true
  enable_anomaly_detection: true
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Fix Python path
   export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
   ```

2. **Redis Connection Error**

   ```bash
   # Start Redis
   docker run -d --name redis -p 6379:6379 redis:7-alpine
   ```

3. **GPU Support (Optional)**

   ```bash
   # Install CUDA-enabled PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Memory Issues**

   ```python
   # Reduce model size in config
   config = OptimizationConfig(
       use_transformer=True,
       transformer_hidden_dim=256,  # Reduce from 512
       transformer_num_layers=4     # Reduce from 8
   )
   ```

### Performance Tuning

```bash
# Enable performance optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# For production
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
```

## 📚 API Documentation

### Interactive API Docs

- **Swagger UI:** <http://localhost:8000/api/docs>
- **ReDoc:** <http://localhost:8000/api/redoc>

### WebSocket API

```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};
```

## 🎓 Learning Resources

### Documentation

- [Architecture Guide](docs/developer_guide/architecture.md)
- [Implementation Details](docs/developer_guide/implementation.md)
- [User Guide](docs/user_guide/introduction.md)

### Examples

- [Basic Optimization](examples/basic_optimization.py)
- [Advanced AI Features](examples/advanced_ai.py)
- [Custom Model Training](examples/custom_training.py)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenRAN Alliance for standards and specifications
- 3GPP for 5G technical specifications
- PyTorch and scikit-learn communities
- FastAPI and Streamlit teams

---

**🏆 The AI-Powered 5G OpenRAN Optimizer is now production-ready with enterprise-grade features, advanced AI capabilities, and comprehensive monitoring!**
