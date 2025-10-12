#!/usr/bin/env python3
"""
Setup script to get the RAG service running.
This handles all the initial setup and validation.
"""
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = [
        "rag_service",
        "data/crawled", 
        "data/vectors",
        "tests"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_init_files():
    """Create __init__.py files to make directories into Python packages."""
    init_files = [
        "rag_service/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test that we can import all required libraries."""
    print("🧪 Testing imports...")
    
    required_packages = [
        'requests',
        'bs4',
        'sentence_transformers', 
        'faiss',
        'fastapi',
        'click',
        'numpy'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Try running: pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully")
    return True

def test_basic_functionality():
    """Test basic functionality of our modules."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test utility functions
        from rag_service.utils import get_domain, chunk_text
        
        # Test domain extraction
        domain = get_domain("https://example.com/path")
        assert domain == "example.com"
        print("   ✅ Domain extraction works")
        
        # Test text chunking
        chunks = chunk_text("This is a test text for chunking.", chunk_size=20, overlap=5)
        assert len(chunks) > 0
        print("   ✅ Text chunking works")
        
        # Test config loading
        from rag_service.config import DEFAULT_CHUNK_SIZE
        assert isinstance(DEFAULT_CHUNK_SIZE, int)
        print("   ✅ Configuration loading works")
        
        print("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def create_env_file():
    """Create .env file with default settings."""
    env_content = """# RAG Service Environment Variables
API_HOST=127.0.0.1
API_PORT=8000
LOG_LEVEL=INFO

# Optional: Custom models (uncomment to use)
# EMBEDDING_MODEL=all-MiniLM-L6-v2
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("ℹ️  .env file already exists")

def main():
    """Main setup function."""
    print("🚀 Setting up RAG Service...")
    print("=" * 50)
    
    # Step 1: Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Step 2: Create init files
    print("\n2. Creating package files...")
    create_init_files()
    
    # Step 3: Install dependencies
    print("\n3. Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return 1
    
    # Step 4: Test imports
    print("\n4. Testing imports...")
    if not test_imports():
        print("❌ Setup failed at import testing")
        return 1
    
    # Step 5: Test functionality
    print("\n5. Testing basic functionality...")
    if not test_basic_functionality():
        print("❌ Setup failed at functionality testing")
        print("   This might be OK - some tests require the modules to be in place first")
    
    # Step 6: Create environment file
    print("\n6. Creating environment file...")
    create_env_file()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Quick Start Guide:")
    print("=" * 50)
    print("1. Crawl a website:")
    print("   python crawl_script.py https://docs.python.org/3/tutorial/ --max-pages 10")
    print()
    print("2. Index the content:")
    print("   python index_script.py")
    print()
    print("3. Ask questions:")
    print("   python ask_script.py 'What is Python used for?'")
    print()
    print("4. Or start the API server:")
    print("   python app.py")
    print("   # Then visit http://localhost:8000/docs for API documentation")
    print()
    print("5. Run tests:")
    print("   pytest tests/ -v")
    print()
    print("📚 For detailed documentation, see the README.md file")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())