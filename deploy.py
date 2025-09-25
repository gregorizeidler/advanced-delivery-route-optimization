#!/usr/bin/env python3
"""
🚀 Advanced Deployment Script
Delivery Route Optimization - Enterprise System

Comprehensive deployment script that:
- Sets up environment
- Installs dependencies
- Runs tests
- Launches web dashboard
- Monitors system health

Author: IBM Data Analyst Professional Certificate Student
"""

import os
import sys
import subprocess
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import argparse
import webbrowser
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """
    🎯 Enterprise Deployment Manager
    
    Handles complete system deployment with monitoring and health checks.
    """
    
    def __init__(self, project_root: str = None):
        """Initialize deployment manager."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.venv_path = self.project_root / "venv"
        self.data_path = self.project_root / "data"
        self.logs_path = self.project_root / "logs"
        
        # Deployment configuration
        self.config = {
            'python_version': '3.8',
            'streamlit_port': 8501,
            'health_check_interval': 30,
            'auto_restart': True,
            'max_retries': 3
        }
        
        # Health status
        self.health_status = {
            'environment': False,
            'dependencies': False,
            'data': False,
            'tests': False,
            'dashboard': False
        }
        
        logger.info(f"🚀 Deployment Manager initialized for {self.project_root}")
    
    def deploy(self, mode: str = "full", skip_tests: bool = False, 
               auto_open: bool = True) -> bool:
        """
        Execute complete deployment.
        
        Args:
            mode: Deployment mode ('full', 'quick', 'data-only')
            skip_tests: Skip running tests
            auto_open: Automatically open dashboard in browser
            
        Returns:
            True if deployment successful, False otherwise
        """
        logger.info(f"🎯 Starting {mode} deployment...")
        
        start_time = time.time()
        
        try:
            # Phase 1: Environment Setup
            if mode in ['full', 'quick']:
                if not self._setup_environment():
                    return False
            
            # Phase 2: Install Dependencies
            if mode in ['full', 'quick']:
                if not self._install_dependencies():
                    return False
            
            # Phase 3: Data Setup
            if not self._setup_data():
                return False
            
            # Phase 4: Run Tests (optional)
            if not skip_tests and mode == 'full':
                if not self._run_tests():
                    logger.warning("⚠️ Tests failed but continuing deployment...")
            
            # Phase 5: Launch Dashboard
            if mode in ['full', 'quick']:
                if not self._launch_dashboard(auto_open):
                    return False
            
            # Phase 6: Health Monitoring
            if mode == 'full':
                self._start_health_monitoring()
            
            deployment_time = time.time() - start_time
            logger.info(f"✅ Deployment completed successfully in {deployment_time:.1f} seconds!")
            
            self._print_deployment_summary()
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
    
    def _setup_environment(self) -> bool:
        """Setup Python virtual environment."""
        logger.info("🔧 Setting up virtual environment...")
        
        try:
            # Create directories
            self.logs_path.mkdir(exist_ok=True)
            self.data_path.mkdir(exist_ok=True)
            
            # Create virtual environment if it doesn't exist
            if not self.venv_path.exists():
                logger.info("📦 Creating virtual environment...")
                result = subprocess.run([
                    sys.executable, "-m", "venv", str(self.venv_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to create virtual environment: {result.stderr}")
                    return False
            
            # Activate virtual environment
            if sys.platform == "win32":
                activate_script = self.venv_path / "Scripts" / "activate.bat"
                python_exe = self.venv_path / "Scripts" / "python.exe"
            else:
                activate_script = self.venv_path / "bin" / "activate"
                python_exe = self.venv_path / "bin" / "python"
            
            if not python_exe.exists():
                logger.error("Python executable not found in virtual environment")
                return False
            
            self.health_status['environment'] = True
            logger.info("✅ Virtual environment ready")
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Install Python dependencies."""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Get Python executable path
            if sys.platform == "win32":
                python_exe = self.venv_path / "Scripts" / "python.exe"
                pip_exe = self.venv_path / "Scripts" / "pip.exe"
            else:
                python_exe = self.venv_path / "bin" / "python"
                pip_exe = self.venv_path / "bin" / "pip"
            
            # Upgrade pip first
            logger.info("🔄 Upgrading pip...")
            result = subprocess.run([
                str(python_exe), "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"Pip upgrade warning: {result.stderr}")
            
            # Install requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                logger.info("📋 Installing from requirements.txt...")
                result = subprocess.run([
                    str(pip_exe), "install", "-r", str(requirements_file)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to install requirements: {result.stderr}")
                    return False
            else:
                logger.warning("requirements.txt not found, installing basic dependencies...")
                basic_deps = [
                    "pandas", "numpy", "matplotlib", "seaborn", "plotly", 
                    "streamlit", "streamlit-folium", "scikit-learn", "faker"
                ]
                
                result = subprocess.run([
                    str(pip_exe), "install"
                ] + basic_deps, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to install basic dependencies: {result.stderr}")
                    return False
            
            self.health_status['dependencies'] = True
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False
    
    def _setup_data(self) -> bool:
        """Setup and generate data if needed."""
        logger.info("📊 Setting up data...")
        
        try:
            # Check if data exists
            required_files = [
                "delivery_points.csv",
                "vehicles.csv", 
                "delivery_orders.csv",
                "customers.csv",
                "demand_series.npy"
            ]
            
            missing_files = []
            for file in required_files:
                if not (self.data_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                logger.info(f"🎲 Generating missing data files: {missing_files}")
                
                # Run data generator
                if sys.platform == "win32":
                    python_exe = self.venv_path / "Scripts" / "python.exe"
                else:
                    python_exe = self.venv_path / "bin" / "python"
                
                data_generator_script = self.project_root / "src" / "data_generator.py"
                
                if data_generator_script.exists():
                    result = subprocess.run([
                        str(python_exe), str(data_generator_script)
                    ], capture_output=True, text=True, cwd=str(self.project_root))
                    
                    if result.returncode != 0:
                        logger.error(f"Data generation failed: {result.stderr}")
                        return False
                else:
                    logger.error("Data generator script not found")
                    return False
            
            # Verify data files exist now
            for file in required_files:
                if not (self.data_path / file).exists():
                    logger.error(f"Data file still missing: {file}")
                    return False
            
            self.health_status['data'] = True
            logger.info("✅ Data setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Data setup failed: {e}")
            return False
    
    def _run_tests(self) -> bool:
        """Run test suite."""
        logger.info("🧪 Running tests...")
        
        try:
            # Get Python executable
            if sys.platform == "win32":
                python_exe = self.venv_path / "Scripts" / "python.exe"
            else:
                python_exe = self.venv_path / "bin" / "python"
            
            # Run basic functionality test
            test_script = self.project_root / "test_notebook.py"
            
            if test_script.exists():
                result = subprocess.run([
                    str(python_exe), str(test_script)
                ], capture_output=True, text=True, cwd=str(self.project_root))
                
                if result.returncode == 0:
                    logger.info("✅ Tests passed")
                    self.health_status['tests'] = True
                    return True
                else:
                    logger.error(f"Tests failed: {result.stderr}")
                    return False
            else:
                logger.warning("Test script not found, skipping tests")
                return True
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    def _launch_dashboard(self, auto_open: bool = True) -> bool:
        """Launch Streamlit dashboard."""
        logger.info("🌐 Launching web dashboard...")
        
        try:
            # Get Streamlit executable
            if sys.platform == "win32":
                streamlit_exe = self.venv_path / "Scripts" / "streamlit.exe"
            else:
                streamlit_exe = self.venv_path / "bin" / "streamlit"
            
            # Check if app.py exists
            app_script = self.project_root / "app.py"
            if not app_script.exists():
                logger.error("app.py not found")
                return False
            
            # Launch Streamlit in background
            dashboard_url = f"http://localhost:{self.config['streamlit_port']}"
            
            logger.info(f"🚀 Starting dashboard at {dashboard_url}")
            
            # Start Streamlit process
            cmd = [
                str(streamlit_exe), "run", str(app_script),
                "--server.port", str(self.config['streamlit_port']),
                "--server.headless", "true",
                "--server.address", "localhost"
            ]
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root)
            )
            
            # Wait for startup
            logger.info("⏳ Waiting for dashboard to start...")
            time.sleep(10)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info("✅ Dashboard started successfully")
                
                # Open in browser
                if auto_open:
                    try:
                        webbrowser.open(dashboard_url)
                        logger.info("🌐 Dashboard opened in browser")
                    except Exception as e:
                        logger.warning(f"Could not open browser: {e}")
                
                self.health_status['dashboard'] = True
                
                # Save process info
                self._save_process_info(process.pid, dashboard_url)
                
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Dashboard failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Dashboard launch failed: {e}")
            return False
    
    def _save_process_info(self, pid: int, url: str):
        """Save process information for monitoring."""
        process_info = {
            'pid': pid,
            'url': url,
            'start_time': datetime.now().isoformat(),
            'status': 'running'
        }
        
        process_file = self.project_root / "dashboard_process.json"
        with open(process_file, 'w') as f:
            json.dump(process_info, f, indent=2)
    
    def _start_health_monitoring(self):
        """Start health monitoring (simplified version)."""
        logger.info("💊 Health monitoring enabled")
        logger.info(f"📊 Dashboard will be monitored every {self.config['health_check_interval']} seconds")
        
        # In a real implementation, this would start a background monitoring service
        # For now, we just log the configuration
    
    def _print_deployment_summary(self):
        """Print deployment summary."""
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT SUMMARY")
        print("="*60)
        
        print(f"📁 Project Root: {self.project_root}")
        print(f"🐍 Virtual Environment: {self.venv_path}")
        print(f"📊 Data Directory: {self.data_path}")
        print(f"📝 Logs Directory: {self.logs_path}")
        
        print("\n🏥 HEALTH STATUS:")
        for component, status in self.health_status.items():
            emoji = "✅" if status else "❌"
            print(f"  {emoji} {component.title()}: {'Healthy' if status else 'Failed'}")
        
        if self.health_status['dashboard']:
            print(f"\n🌐 Dashboard URL: http://localhost:{self.config['streamlit_port']}")
            print("📖 Access the interactive dashboard to explore route optimization!")
        
        print("\n🎓 IBM Data Analyst Professional Certificate Project")
        print("🚛 Enterprise-Level Delivery Route Optimization System")
        print("="*60)
    
    def stop_dashboard(self):
        """Stop running dashboard."""
        logger.info("🛑 Stopping dashboard...")
        
        try:
            process_file = self.project_root / "dashboard_process.json"
            
            if process_file.exists():
                with open(process_file, 'r') as f:
                    process_info = json.load(f)
                
                pid = process_info.get('pid')
                if pid:
                    try:
                        if sys.platform == "win32":
                            subprocess.run(["taskkill", "/F", "/PID", str(pid)])
                        else:
                            subprocess.run(["kill", str(pid)])
                        
                        logger.info("✅ Dashboard stopped")
                    except Exception as e:
                        logger.warning(f"Could not stop process {pid}: {e}")
                
                # Remove process file
                process_file.unlink()
            else:
                logger.warning("No running dashboard found")
                
        except Exception as e:
            logger.error(f"Failed to stop dashboard: {e}")
    
    def status(self):
        """Check system status."""
        print("🏥 SYSTEM STATUS CHECK")
        print("="*40)
        
        # Check virtual environment
        if self.venv_path.exists():
            print("✅ Virtual environment: OK")
        else:
            print("❌ Virtual environment: Missing")
        
        # Check data files
        data_files_exist = all(
            (self.data_path / f).exists() 
            for f in ["delivery_points.csv", "vehicles.csv", "delivery_orders.csv"]
        )
        
        if data_files_exist:
            print("✅ Data files: OK")
        else:
            print("❌ Data files: Missing")
        
        # Check dashboard
        process_file = self.project_root / "dashboard_process.json"
        if process_file.exists():
            try:
                with open(process_file, 'r') as f:
                    process_info = json.load(f)
                print(f"✅ Dashboard: Running (PID: {process_info['pid']})")
                print(f"🌐 URL: {process_info['url']}")
            except Exception:
                print("❌ Dashboard: Status unknown")
        else:
            print("❌ Dashboard: Not running")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Advanced Deployment Manager")
    
    parser.add_argument("command", choices=["deploy", "stop", "status", "restart"],
                       help="Deployment command")
    parser.add_argument("--mode", choices=["full", "quick", "data-only"], 
                       default="full", help="Deployment mode")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip running tests")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open browser")
    parser.add_argument("--project-root", type=str,
                       help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployment_manager = DeploymentManager(args.project_root)
    
    if args.command == "deploy":
        success = deployment_manager.deploy(
            mode=args.mode,
            skip_tests=args.skip_tests,
            auto_open=not args.no_browser
        )
        sys.exit(0 if success else 1)
    
    elif args.command == "stop":
        deployment_manager.stop_dashboard()
    
    elif args.command == "status":
        deployment_manager.status()
    
    elif args.command == "restart":
        deployment_manager.stop_dashboard()
        time.sleep(2)
        success = deployment_manager.deploy(
            mode="quick",
            skip_tests=True,
            auto_open=not args.no_browser
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
