# 🚀 **AI-Powered 5G Open RAN Optimizer - Installation Guide**

[![Installation Status](https://img.shields.io/badge/Installation-One%20Command-green.svg)]()
[![Platform Support](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue.svg)]()
[![Cloud Ready](https://img.shields.io/badge/Cloud-Azure%20Native-0078d4.svg)]()
[![Docker Support](https://img.shields.io/badge/Docker-Containerized-2496ed.svg)]()

> **⚡ Get started with the world's most advanced AI-powered 5G network optimizer in minutes!**

📋 **Quick Navigation:**

- [🚀 Quick Start (One Command)](#-quick-start-one-command)
- [💻 Local Development Setup](#-local-development-setup)
- [🐳 Docker Deployment](#-docker-deployment)
- [☁️ Azure Cloud Deployment](#️-azure-cloud-deployment)
- [🔧 Advanced Configuration](#-advanced-configuration)
- [✅ Verification & Testing](#-verification--testing)
- [🔍 Troubleshooting](#-troubleshooting)

---

## 🎯 **Installation Options**

Choose your preferred installation method:

| Method | Time to Deploy | Use Case | Complexity |
|--------|----------------|----------|------------|
| [🚀 One-Command Cloud](#-quick-start-one-command) | **2 minutes** | Production, Evaluation | ⭐ Easy |
| [💻 Local Development](#-local-development-setup) | **5 minutes** | Development, Testing | ⭐⭐ Medium |
| [🐳 Docker Containers](#-docker-deployment) | **3 minutes** | Portable, Isolated | ⭐⭐ Medium |
| [☁️ Manual Azure Setup](#️-azure-cloud-deployment) | **10 minutes** | Custom Configuration | ⭐⭐⭐ Advanced |

---

## 🚀 **Quick Start (One Command)**

### **☁️ Azure Cloud Deployment**

Deploy the complete AI-powered 5G optimizer with Azure infrastructure in one command:

```bash
# Prerequisites: Azure CLI + Azure Developer CLI
az login
azd auth login

# Clone and deploy (complete setup in ~2 minutes)
git clone https://github.com/sachin-deshik-10/5G_AI_POWERED_ORAN.git
cd 5G_AI_POWERED_ORAN
azd up
```

**🎉 That's it!** The system will automatically:

- ✅ Provision Azure infrastructure (Container Apps, Cosmos DB, OpenAI, etc.)
- ✅ Build and deploy all microservices
- ✅ Configure AI models and security
- ✅ Set up monitoring and observability
- ✅ Provide you with endpoints and dashboard URL

---

## 💻 **Local Development Setup**

### **📋 Prerequisites**

#### **Required Software**

| Software | Version | Purpose | Installation |
|----------|---------|---------|--------------|
| **Python** | 3.11+ | Core runtime | [Download Python](https://www.python.org/downloads/) |
| **Node.js** | 18+ | Dashboard frontend | [Download Node.js](https://nodejs.org/) |
| **Git** | Latest | Version control | [Download Git](https://git-scm.com/) |
| **Docker** | Latest | Containerization | [Download Docker](https://www.docker.com/) |

#### **Optional Tools (Recommended)**

| Tool | Purpose | Installation |
|------|---------|--------------|
| **Azure CLI** | Cloud deployment | `curl -sL https://aka.ms/InstallAzureCLIDeb \| sudo bash` |
| **Azure Developer CLI** | One-command deployment | `curl -fsSL https://aka.ms/install-azd.sh \| bash` |
| **VS Code** | Development IDE | [Download VS Code](https://code.visualstudio.com/) |

### **⚙️ System Requirements**

#### **Minimum Requirements**

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **OS** | Windows 10+, macOS 10.15+, Ubuntu 20.04+ | Cross-platform support |
| **RAM** | 8 GB | For basic operation |
| **CPU** | 4 cores, 2.5 GHz | Intel/AMD x64 or ARM64 |
| **Storage** | 20 GB free space | For models and data |
| **Network** | Broadband internet | For Azure services |

#### **Recommended Requirements**

| Component | Requirement | Performance Benefit |
|-----------|-------------|-------------------|
| **RAM** | 16+ GB | Faster AI model training |
| **CPU** | 8+ cores, 3.0+ GHz | Parallel processing |
| **GPU** | NVIDIA GTX 1660+ | Accelerated AI inference |
| **Storage** | SSD, 50+ GB | Faster I/O operations |

### **📦 Step-by-Step Installation**

#### **1. Clone the Repository**

```bash
# Clone with all submodules
git clone --recursive https://github.com/sachin-deshik-10/5G_AI_POWERED_ORAN.git
cd 5G_AI_POWERED_ORAN

# Verify clone integrity
git status
```

#### **2. Create Python Environment**

**Option A: Using venv (Recommended)**

```bash
# Create virtual environment
python -m venv network_optimizer_env

# Activate environment
# Windows
network_optimizer_env\Scripts\activate
# Linux/macOS
source network_optimizer_env/bin/activate

# Verify activation
which python  # Should point to virtual environment
```

**Option B: Using conda**

```bash
# Create conda environment
conda create -n network_optimizer python=3.11 -y
conda activate network_optimizer

# Verify activation
conda info --envs
```

#### **3. Install Dependencies**

**Core Dependencies:**

```bash
# Upgrade pip and install core packages
python -m pip install --upgrade pip setuptools wheel

# Install main dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(torch|transformers|azure|fastapi)"
```

**Optional Dependencies for Advanced Features:**

```bash
# AI/ML Advanced Features
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Quantum Computing (Optional)
pip install qiskit[all] cirq tensorflow-quantum

# Neuromorphic Computing (Optional)
pip install nengo loihi-api

# Geospatial Visualization
pip install folium plotly geopandas

# Development Tools
pip install pytest black flake8 mypy pre-commit
```

#### **4. Configure Environment Variables**

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env  # or code .env for VS Code
```

**Example .env configuration:**

```bash
# Core Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true

# AI Model Configuration
MODEL_PATH=./models/
CACHE_SIZE=1000
ENABLE_GPU=false

# Azure Configuration (Optional for local development)
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_OPENAI_ENDPOINT=your-openai-endpoint
AZURE_OPENAI_KEY=your-openai-key

# Database Configuration
DATABASE_URL=sqlite:///./network_optimizer.db
REDIS_URL=redis://localhost:6379

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
```

#### **5. Initialize Database & Models**

```bash
# Initialize database schema
python scripts/init_database.py

# Download pre-trained models
python scripts/download_models.py

# Verify setup
python scripts/verify_installation.py
```

#### **6. Start Development Servers**

**Terminal 1 - API Server:**

```bash
# Start FastAPI backend
python api/api_server.py

# Server will start at: http://localhost:8000
# API docs available at: http://localhost:8000/docs
```

**Terminal 2 - Dashboard:**

```bash
# Start Streamlit dashboard
streamlit run dashboard/real_time_monitor.py

# Dashboard will start at: http://localhost:8501
```

**Terminal 3 - AI Processing Engine:**

```bash
# Start AI optimization engine
python src/main.py --mode development

# Monitor logs for initialization status
```

---

## 🐳 **Docker Deployment**

### **🚀 Quick Docker Setup**

```bash
# Build and start all services
docker-compose up -d --build

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f

# Access services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - Documentation: http://localhost:8000/docs
```

### **📋 Individual Container Deployment**

**API Server:**

```bash
# Build API container
docker build -f Dockerfile.api -t 5g-oran-api:latest .

# Run API container
docker run -d \
  --name 5g-oran-api \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -v $(pwd)/data:/app/data \
  5g-oran-api:latest
```

**Dashboard:**

```bash
# Build dashboard container
docker build -f Dockerfile.dashboard -t 5g-oran-dashboard:latest .

# Run dashboard container
docker run -d \
  --name 5g-oran-dashboard \
  -p 8501:8501 \
  -e API_ENDPOINT=http://5g-oran-api:8000 \
  --link 5g-oran-api:api \
  5g-oran-dashboard:latest
```

### **🔧 Production Docker Configuration**

**docker-compose.prod.yml:**

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - WORKERS=4
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    environment:
      - API_ENDPOINT=http://api:8000
    ports:
      - "8501:8501"
    depends_on:
      - api
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

---

## ☁️ **Azure Cloud Deployment**

### **🎯 Prerequisites for Azure Deployment**

1. **Azure Account**: [Create free account](https://azure.microsoft.com/free/)
2. **Azure CLI**: [Installation guide](https://docs.microsoft.com/cli/azure/install-azure-cli)
3. **Azure Developer CLI**: [Installation guide](https://docs.microsoft.com/azure/developer/azure-developer-cli/install-azd)

### **🔧 Manual Azure Setup**

#### **1. Resource Group Creation**

```bash
# Set variables
RESOURCE_GROUP="rg-5g-oran-optimizer"
LOCATION="eastus"
SUBSCRIPTION_ID="your-subscription-id"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION \
  --subscription $SUBSCRIPTION_ID
```

#### **2. Infrastructure Deployment**

```bash
# Deploy infrastructure using Bicep
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file infra/main.bicep \
  --parameters @infra/main.parameters.json \
  --parameters environmentName=production

# Verify deployment
az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name main \
  --query "properties.provisioningState"
```

#### **3. Application Deployment**

```bash
# Build and push container images
az acr build --registry your-registry --image 5g-oran-api:latest -f Dockerfile.api .
az acr build --registry your-registry --image 5g-oran-dashboard:latest -f Dockerfile.dashboard .

# Deploy to Container Apps
az containerapp update \
  --name api-container-app \
  --resource-group $RESOURCE_GROUP \
  --image your-registry.azurecr.io/5g-oran-api:latest
```

### **🔐 Security Configuration**

```bash
# Configure Key Vault
az keyvault create \
  --name kv-5g-oran-optimizer \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Store secrets
az keyvault secret set \
  --vault-name kv-5g-oran-optimizer \
  --name "openai-api-key" \
  --value "your-openai-key"

# Configure managed identity
az identity create \
  --name id-5g-oran-optimizer \
  --resource-group $RESOURCE_GROUP
```

---

## 🔧 **Advanced Configuration**

### **🎛️ AI Model Configuration**

**config/advanced_config.yaml:**

```yaml
# AI Configuration
ai_models:
  cognitive_engine:
    model_type: "transformer"
    model_size: "large"
    quantization: true
    batch_size: 32
    max_sequence_length: 512
    
  edge_ai:
    deployment_target: "onnx"
    optimization_level: "O3"
    precision: "fp16"
    cache_size: "1GB"
    
  security_ai:
    threat_detection:
      sensitivity: "high"
      response_time: "5s"
      ml_models: ["isolation_forest", "lstm_autoencoder"]
    
  quantum_optimization:
    enabled: true
    backend: "qiskit"
    noise_model: "ibm_brisbane"
    shots: 1024

# Performance Configuration
performance:
  max_workers: 8
  memory_limit: "8GB"
  cache_ttl: 3600
  batch_processing: true
  
# Monitoring Configuration
monitoring:
  metrics_interval: 10
  log_level: "INFO"
  trace_sampling: 0.1
  alerts_enabled: true
```

### **🔬 Quantum Computing Setup (Optional)**

```bash
# Install Qiskit and IBM Quantum dependencies
pip install qiskit[all] qiskit-ibm-runtime

# Configure IBM Quantum account
python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_IBM_QUANTUM_TOKEN')
"

# Test quantum backend connection
python scripts/test_quantum_connection.py
```

### **🧠 Neuromorphic Computing Setup (Optional)**

```bash
# Install Intel Loihi dependencies (if available)
pip install nxsdk nengo-loihi

# Configure neuromorphic simulation
python scripts/setup_neuromorphic.py

# Test neuromorphic processing
python scripts/test_neuromorphic.py
```

---

## ✅ **Verification & Testing**

### **🧪 Comprehensive System Testing**

```bash
# Run complete test suite
python -m pytest tests/ -v --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v           # Unit tests
python -m pytest tests/integration/ -v    # Integration tests
python -m pytest tests/performance/ -v    # Performance tests

# Run AI model validation
python scripts/validate_models.py

# Run end-to-end system test
python scripts/e2e_test.py
```

### **📊 Performance Benchmarking**

```bash
# Run performance benchmarks
python scripts/benchmark_system.py

# Generate performance report
python scripts/generate_performance_report.py

# Test API performance
python scripts/api_load_test.py --concurrent-users 100 --duration 60s
```

### **🔍 Health Checks**

```bash
# System health check
curl http://localhost:8000/health

# Detailed diagnostics
curl http://localhost:8000/diagnostics

# AI model status
curl http://localhost:8000/models/status

# Database connectivity
curl http://localhost:8000/db/health
```

---

## 🔍 **Troubleshooting**

### **❗ Common Issues & Solutions**

#### **Installation Issues**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Python Version** | `ModuleNotFoundError` | Upgrade to Python 3.11+ |
| **Memory Error** | System freezes during installation | Increase virtual memory, close other applications |
| **Network Timeout** | Package download fails | Use `pip install --timeout 300` |
| **Permission Error** | Access denied during installation | Use `pip install --user` or run as administrator |

#### **Runtime Issues**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Port Already in Use** | `Address already in use` | Change port in config or kill process: `lsof -ti:8000 \| xargs kill` |
| **Database Connection** | Connection refused | Start database service: `systemctl start postgresql` |
| **AI Model Loading** | Model not found error | Run: `python scripts/download_models.py` |
| **GPU Not Detected** | CUDA initialization failed | Install CUDA toolkit and restart |

#### **Performance Issues**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Slow Response** | High latency | Enable caching: `ENABLE_CACHE=true` |
| **High Memory Usage** | System slows down | Reduce batch size in config |
| **CPU Bottleneck** | 100% CPU usage | Increase worker count or scale horizontally |

### **🛠️ Debug Commands**

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Check system resources
python scripts/check_system_resources.py

# Validate configuration
python scripts/validate_config.py

# Test network connectivity
python scripts/test_connectivity.py

# Monitor system performance
python scripts/monitor_performance.py --duration 300
```

### **📞 Getting Help**

If you encounter issues not covered here:

1. **📖 Check Documentation**: [docs/](../README.md)
2. **🐛 Search Issues**: [GitHub Issues](https://github.com/sachin-deshik-10/5G_AI_POWERED_ORAN/issues)
3. **💬 Join Discord**: [Community Support](https://discord.gg/5g-oran)
4. **📧 Email Support**: <support@5g-oran-optimizer.ai>
5. **📋 Create Issue**: [New Issue Template](https://github.com/sachin-deshik-10/5G_AI_POWERED_ORAN/issues/new)

---

## 🎉 **Next Steps**

### **📚 After Installation**

1. **📖 Read Documentation**:
   - [User Guide](../user_guide/README.md)
   - [API Reference](../api/README.md)
   - [Developer Guide](../developer_guide/README.md)

2. **🚀 Try Examples**:

   ```bash
   python examples/basic_optimization.py
   python examples/advanced_visualization.py
   python examples/quantum_optimization.py
   ```

3. **🔧 Customize Configuration**:
   - Edit `config/advanced_config.yaml`
   - Set up monitoring dashboards
   - Configure AI model parameters

4. **🤝 Join Community**:
   - [Contribute](../../CONTRIBUTING.md)
   - [Code of Conduct](../../CODE_OF_CONDUCT.md)
   - [Governance](../../GOVERNANCE.md)

### **🚀 Production Deployment Checklist**

- [ ] **Security**: Configure HTTPS, API keys, firewall rules
- [ ] **Monitoring**: Set up alerts, logging, performance monitoring
- [ ] **Scaling**: Configure auto-scaling, load balancing
- [ ] **Backup**: Set up automated backups for models and data
- [ ] **Documentation**: Create runbooks and operational procedures
- [ ] **Testing**: Run full test suite and load testing
- [ ] **Compliance**: Ensure regulatory compliance (GDPR, SOC2, etc.)

---

**🎯 Ready to revolutionize 5G networks with AI? You're all set!**

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template)

*For additional support, visit our [Community Discord](https://discord.gg/5g-oran) or check our [Troubleshooting Guide](troubleshooting.md).*
