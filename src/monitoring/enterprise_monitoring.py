"""
Enterprise Monitoring and Alerting System
Comprehensive observability for 5G O-RAN Optimizer with Prometheus, Grafana integration
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertChannel(Enum):
    """Alert notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"

@dataclass
class MetricThreshold:
    """Metric threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str  # 'gt', 'lt', 'eq'
    time_window_minutes: int = 5
    evaluation_frequency_seconds: int = 30

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    source: str
    tags: Dict[str, str]
    resolved: bool = False
    acknowledged: bool = False

@dataclass
class MonitoringConfig:
    """Monitoring system configuration"""
    prometheus_port: int = 8001
    metrics_collection_interval: int = 10  # seconds
    alert_evaluation_interval: int = 30  # seconds
    retention_days: int = 30
    enable_email_alerts: bool = True
    enable_slack_alerts: bool = True
    enable_webhook_alerts: bool = True
    
    # Alert channels configuration
    email_config: Dict[str, str] = None
    slack_config: Dict[str, str] = None
    webhook_config: Dict[str, str] = None

class EnterpriseMonitoringSystem:
    """Enterprise-grade monitoring and alerting system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.registry = CollectorRegistry()
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.metric_history: Dict[str, List[Dict]] = {}
        
        # Initialize Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Metric thresholds
        self.thresholds = self._setup_default_thresholds()
        
        # Alert channels
        self.alert_channels = []
        self._setup_alert_channels()
        
        # Background tasks
        self.monitoring_tasks = []
        
        logging.info("Enterprise Monitoring System initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # System performance metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent', 
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage', 
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # Network performance metrics
        self.network_throughput = Gauge(
            'network_throughput_mbps',
            'Network throughput in Mbps',
            ['direction']  # 'uplink', 'downlink'
        )
        
        self.network_latency = Histogram(
            'network_latency_milliseconds',
            'Network latency in milliseconds',
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self.registry
        )
        
        self.packet_loss = Gauge(
            'network_packet_loss_percent',
            'Packet loss percentage',
            registry=self.registry
        )
        
        # AI/ML metrics
        self.model_inference_time = Histogram(
            'ai_model_inference_time_seconds',
            'AI model inference time',
            ['model_type'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'ai_model_accuracy_score',
            'AI model accuracy score',
            ['model_name'],
            registry=self.registry
        )
        
        self.optimization_requests = Counter(
            'optimization_requests_total',
            'Total optimization requests',
            ['optimization_type', 'status'],
            registry=self.registry
        )
        
        # Business metrics
        self.energy_efficiency = Gauge(
            'energy_efficiency_mbps_per_watt',
            'Energy efficiency in Mbps per Watt',
            registry=self.registry
        )
        
        self.user_satisfaction = Gauge(
            'user_satisfaction_score',
            'User satisfaction score (0-100)',
            registry=self.registry
        )
        
        self.cost_savings = Gauge(
            'cost_savings_percent',
            'Cost savings percentage',
            registry=self.registry
        )
        
        # Security metrics
        self.security_threats = Counter(
            'security_threats_detected_total',
            'Total security threats detected',
            ['threat_type', 'severity'],
            registry=self.registry
        )
        
        self.security_response_time = Histogram(
            'security_response_time_seconds',
            'Security response time in seconds',
            registry=self.registry
        )
    
    def _setup_default_thresholds(self) -> List[MetricThreshold]:
        """Setup default metric thresholds"""
        return [
            MetricThreshold("cpu_usage", 80.0, 90.0, "gt"),
            MetricThreshold("memory_usage", 85.0, 95.0, "gt"),
            MetricThreshold("disk_usage", 80.0, 90.0, "gt"),
            MetricThreshold("network_latency", 10.0, 20.0, "gt"),
            MetricThreshold("packet_loss", 0.1, 1.0, "gt"),
            MetricThreshold("network_throughput", 50.0, 30.0, "lt"),
            MetricThreshold("energy_efficiency", 15.0, 10.0, "lt"),
            MetricThreshold("user_satisfaction", 80.0, 70.0, "lt"),
            MetricThreshold("model_accuracy", 0.85, 0.75, "lt"),
        ]
    
    def _setup_alert_channels(self):
        """Setup alert notification channels"""
        if self.config.enable_email_alerts and self.config.email_config:
            self.alert_channels.append(EmailAlertChannel(self.config.email_config))
        
        if self.config.enable_slack_alerts and self.config.slack_config:
            self.alert_channels.append(SlackAlertChannel(self.config.slack_config))
        
        if self.config.enable_webhook_alerts and self.config.webhook_config:
            self.alert_channels.append(WebhookAlertChannel(self.config.webhook_config))
    
    async def start_monitoring(self):
        """Start monitoring and alerting system"""
        # Start Prometheus metrics server
        start_http_server(self.config.prometheus_port, registry=self.registry)
        logging.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_network_metrics()),
            asyncio.create_task(self._collect_ai_metrics()),
            asyncio.create_task(self._evaluate_alerts()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        logging.info("Monitoring system started successfully")
    
    async def stop_monitoring(self):
        """Stop monitoring system"""
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        logging.info("Monitoring system stopped")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        import psutil
        
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.set(cpu_percent)
                self._store_metric_history("cpu_usage", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.memory_usage.set(memory_percent)
                self._store_metric_history("memory_usage", memory_percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.disk_usage.set(disk_percent)
                self._store_metric_history("disk_usage", disk_percent)
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logging.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(5)
    
    async def _collect_network_metrics(self):
        """Collect network performance metrics"""
        while True:
            try:
                # Simulate network metrics collection
                # In production, integrate with your network monitoring tools
                
                # Throughput (simulated)
                dl_throughput = np.random.normal(85, 15)  # Mbps
                ul_throughput = np.random.normal(45, 10)  # Mbps
                
                self.network_throughput.labels(direction='downlink').set(dl_throughput)
                self.network_throughput.labels(direction='uplink').set(ul_throughput)
                
                self._store_metric_history("dl_throughput", dl_throughput)
                self._store_metric_history("ul_throughput", ul_throughput)
                
                # Latency (simulated)
                latency_ms = np.random.exponential(5)  # ms
                self.network_latency.observe(latency_ms / 1000)  # Convert to seconds
                self._store_metric_history("network_latency", latency_ms)
                
                # Packet loss (simulated)
                packet_loss = max(0, np.random.exponential(0.05))  # %
                self.packet_loss.set(packet_loss)
                self._store_metric_history("packet_loss", packet_loss)
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logging.error(f"Error collecting network metrics: {e}")
                await asyncio.sleep(5)
    
    async def _collect_ai_metrics(self):
        """Collect AI/ML performance metrics"""
        while True:
            try:
                # Model accuracy (simulated)
                accuracy = np.random.normal(0.9, 0.05)
                self.model_accuracy.labels(model_name='network_optimizer').set(accuracy)
                self._store_metric_history("model_accuracy", accuracy)
                
                # Energy efficiency (simulated)
                energy_eff = np.random.normal(20, 3)  # Mbps/W
                self.energy_efficiency.set(energy_eff)
                self._store_metric_history("energy_efficiency", energy_eff)
                
                # User satisfaction (simulated)
                user_sat = np.random.normal(85, 8)  # Score 0-100
                self.user_satisfaction.set(max(0, min(100, user_sat)))
                self._store_metric_history("user_satisfaction", user_sat)
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logging.error(f"Error collecting AI metrics: {e}")
                await asyncio.sleep(5)
    
    def _store_metric_history(self, metric_name: str, value: float):
        """Store metric value in history"""
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append({
            'timestamp': datetime.now(),
            'value': value
        })
        
        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
        self.metric_history[metric_name] = [
            entry for entry in self.metric_history[metric_name]
            if entry['timestamp'] > cutoff_time
        ]
    
    async def _evaluate_alerts(self):
        """Evaluate alert conditions"""
        while True:
            try:
                for threshold in self.thresholds:
                    await self._check_threshold(threshold)
                
                await asyncio.sleep(self.config.alert_evaluation_interval)
                
            except Exception as e:
                logging.error(f"Error evaluating alerts: {e}")
                await asyncio.sleep(5)
    
    async def _check_threshold(self, threshold: MetricThreshold):
        """Check if metric exceeds threshold"""
        metric_name = threshold.metric_name
        
        if metric_name not in self.metric_history:
            return
        
        # Get recent values within time window
        cutoff_time = datetime.now() - timedelta(minutes=threshold.time_window_minutes)
        recent_values = [
            entry['value'] for entry in self.metric_history[metric_name]
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_values:
            return
        
        # Calculate average value in time window
        avg_value = np.mean(recent_values)
        
        # Check thresholds
        critical_triggered = self._check_condition(
            avg_value, threshold.critical_threshold, threshold.comparison_operator
        )
        warning_triggered = self._check_condition(
            avg_value, threshold.warning_threshold, threshold.comparison_operator
        )
        
        alert_id = f"{metric_name}_{threshold.comparison_operator}_{threshold.warning_threshold}"
        
        if critical_triggered:
            await self._trigger_alert(
                alert_id, AlertSeverity.CRITICAL, metric_name, 
                avg_value, threshold.critical_threshold, threshold
            )
        elif warning_triggered:
            await self._trigger_alert(
                alert_id, AlertSeverity.HIGH, metric_name,
                avg_value, threshold.warning_threshold, threshold
            )
        else:
            # Resolve alert if it was active
            await self._resolve_alert(alert_id)
    
    def _check_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Check if condition is met"""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return abs(value - threshold) < 0.01
        return False
    
    async def _trigger_alert(self, alert_id: str, severity: AlertSeverity, 
                           metric_name: str, current_value: float, 
                           threshold_value: float, threshold_config: MetricThreshold):
        """Trigger a new alert or update existing one"""
        
        # Check if alert already exists
        if alert_id in self.active_alerts:
            # Update existing alert
            self.active_alerts[alert_id].current_value = current_value
            self.active_alerts[alert_id].timestamp = datetime.now()
            return
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=f"{metric_name.replace('_', ' ').title()} Threshold Exceeded",
            description=f"{metric_name} is {current_value:.2f}, exceeding threshold of {threshold_value:.2f}",
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=datetime.now(),
            source="monitoring_system",
            tags={
                "metric": metric_name,
                "severity": severity.value,
                "operator": threshold_config.comparison_operator
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alerts.append(alert)
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logging.warning(f"Alert triggered: {alert.title}")
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            
            # Send resolution notification
            await self._send_resolution_notifications(alert)
            
            logging.info(f"Alert resolved: {alert.title}")
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        for channel in self.alert_channels:
            try:
                await channel.send_alert(alert)
            except Exception as e:
                logging.error(f"Failed to send alert via {channel.__class__.__name__}: {e}")
    
    async def _send_resolution_notifications(self, alert: Alert):
        """Send alert resolution notifications"""
        for channel in self.alert_channels:
            try:
                await channel.send_resolution(alert)
            except Exception as e:
                logging.error(f"Failed to send resolution via {channel.__class__.__name__}: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old monitoring data"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(days=self.config.retention_days)
                
                # Clean up resolved alerts
                self.alerts = [
                    alert for alert in self.alerts
                    if not alert.resolved or alert.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score"""
        scores = []
        
        # Check recent metrics
        for metric_name in ["cpu_usage", "memory_usage", "network_latency", "model_accuracy"]:
            if metric_name in self.metric_history:
                recent_values = self.metric_history[metric_name][-10:]  # Last 10 values
                if recent_values:
                    avg_value = np.mean([v['value'] for v in recent_values])
                    
                    # Convert to health score (0-100)
                    if metric_name in ["cpu_usage", "memory_usage"]:
                        score = max(0, 100 - avg_value)  # Lower usage = better
                    elif metric_name == "network_latency":
                        score = max(0, 100 - (avg_value / 20 * 100))  # Lower latency = better
                    elif metric_name == "model_accuracy":
                        score = avg_value * 100  # Higher accuracy = better
                    else:
                        score = 50  # Default neutral score
                    
                    scores.append(score)
        
        return np.mean(scores) if scores else 50.0
    
    def get_active_alerts_summary(self) -> Dict[str, int]:
        """Get summary of active alerts by severity"""
        summary = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.active_alerts.values():
            summary[alert.severity.value] += 1
        
        return summary
    
    def export_metrics_dashboard_config(self) -> Dict[str, Any]:
        """Export Grafana dashboard configuration"""
        return {
            "dashboard": {
                "title": "5G O-RAN Optimizer Monitoring",
                "panels": [
                    {
                        "title": "System Performance",
                        "type": "graph",
                        "metrics": [
                            "system_cpu_usage_percent",
                            "system_memory_usage_percent",
                            "system_disk_usage_percent"
                        ]
                    },
                    {
                        "title": "Network Performance", 
                        "type": "graph",
                        "metrics": [
                            "network_throughput_mbps",
                            "network_latency_milliseconds",
                            "network_packet_loss_percent"
                        ]
                    },
                    {
                        "title": "AI/ML Metrics",
                        "type": "graph", 
                        "metrics": [
                            "ai_model_accuracy_score",
                            "ai_model_inference_time_seconds",
                            "optimization_requests_total"
                        ]
                    }
                ]
            }
        }

class EmailAlertChannel:
    """Email alert notification channel"""
    
    def __init__(self, config: Dict[str, str]):
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = int(config.get('smtp_port', 587))
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', '').split(',')
    
    async def send_alert(self, alert: Alert):
        """Send alert via email"""
        subject = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
Alert Details:
- Severity: {alert.severity.value.upper()}
- Metric: {alert.metric_name}
- Current Value: {alert.current_value:.2f}
- Threshold: {alert.threshold_value:.2f}
- Time: {alert.timestamp}
- Description: {alert.description}

Tags: {', '.join([f'{k}={v}' for k, v in alert.tags.items()])}
        """
        
        await self._send_email(subject, body)
    
    async def send_resolution(self, alert: Alert):
        """Send resolution notification via email"""
        subject = f"[RESOLVED] {alert.title}"
        body = f"Alert has been resolved: {alert.description}"
        await self._send_email(subject, body)
    
    async def _send_email(self, subject: str, body: str):
        """Send email using SMTP"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
        except Exception as e:
            logging.error(f"Failed to send email: {e}")

class SlackAlertChannel:
    """Slack alert notification channel"""
    
    def __init__(self, config: Dict[str, str]):
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
    
    async def send_alert(self, alert: Alert):
        """Send alert to Slack"""
        color = {
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.HIGH: "warning", 
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "good",
            AlertSeverity.INFO: "good"
        }.get(alert.severity, "warning")
        
        payload = {
            "channel": self.channel,
            "attachments": [{
                "color": color,
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Current Value", "value": f"{alert.current_value:.2f}", "short": True},
                    {"title": "Threshold", "value": f"{alert.threshold_value:.2f}", "short": True},
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True}
                ],
                "timestamp": int(alert.timestamp.timestamp())
            }]
        }
        
        await self._send_slack_message(payload)
    
    async def send_resolution(self, alert: Alert):
        """Send resolution notification to Slack"""
        payload = {
            "channel": self.channel,
            "attachments": [{
                "color": "good",
                "title": f"✅ Resolved: {alert.title}",
                "text": f"Alert has been resolved: {alert.description}"
            }]
        }
        
        await self._send_slack_message(payload)
    
    async def _send_slack_message(self, payload: Dict):
        """Send message to Slack webhook"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status != 200:
                        logging.error(f"Slack webhook returned status {response.status}")
        except Exception as e:
            logging.error(f"Failed to send Slack message: {e}")

class WebhookAlertChannel:
    """Generic webhook alert notification channel"""
    
    def __init__(self, config: Dict[str, str]):
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {})
    
    async def send_alert(self, alert: Alert):
        """Send alert via webhook"""
        payload = {
            "event_type": "alert_triggered",
            "alert": asdict(alert)
        }
        
        await self._send_webhook(payload)
    
    async def send_resolution(self, alert: Alert):
        """Send resolution via webhook"""
        payload = {
            "event_type": "alert_resolved",
            "alert": asdict(alert)
        }
        
        await self._send_webhook(payload)
    
    async def _send_webhook(self, payload: Dict):
        """Send webhook request"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url, 
                    json=payload, 
                    headers=self.headers
                ) as response:
                    if response.status not in [200, 201, 202]:
                        logging.error(f"Webhook returned status {response.status}")
        except Exception as e:
            logging.error(f"Failed to send webhook: {e}")

# Example usage and configuration
if __name__ == "__main__":
    # Configure monitoring system
    monitoring_config = MonitoringConfig(
        prometheus_port=8001,
        metrics_collection_interval=10,
        alert_evaluation_interval=30,
        enable_email_alerts=True,
        email_config={
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': '587',
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',
            'from_email': 'your-email@gmail.com',
            'to_emails': 'admin@yourcompany.com,ops@yourcompany.com'
        },
        enable_slack_alerts=True,
        slack_config={
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'channel': '#5g-alerts'
        }
    )
    
    # Start monitoring system
    async def main():
        monitoring_system = EnterpriseMonitoringSystem(monitoring_config)
        
        try:
            await monitoring_system.start_monitoring()
            
            # Keep running
            while True:
                health_score = monitoring_system.get_system_health_score()
                alerts_summary = monitoring_system.get_active_alerts_summary()
                
                print(f"System Health Score: {health_score:.1f}/100")
                print(f"Active Alerts: {alerts_summary}")
                
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            print("Shutting down monitoring system...")
            await monitoring_system.stop_monitoring()
    
    asyncio.run(main())
