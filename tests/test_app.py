#!/usr/bin/env python3
"""
Author: Cameron Murphy (Student ID: 1049678, GitHub: 0x1049678II)
Date: June 9th 2025

Test script for Indra Travel Bot.
"""

import os
import sys

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    try:
        from flask import Flask
        from chatterbot import ChatBot
        from config.settings import Config
        from chatbot.indra import IndraBot
        print("SUCCESS: All imports working")
        return True
    except Exception as e:
        print(f"FAILED: Import error - {e}")
        return False

def test_flask_app():
    """Test Flask app creation and template access."""
    print("Testing Flask app...")
    try:
        from app import app
        
        # Test app creation
        if not app:
            print("FAILED: Flask app not created")
            return False
        
        # Test template exists
        template_path = os.path.join(app.template_folder or 'templates', 'chat.html')
        if not os.path.exists(template_path):
            print(f"FAILED: Template not found at {template_path}")
            return False
        
        # Test static files exist
        static_path = os.path.join(app.static_folder or 'static', 'css', 'style.css')
        if not os.path.exists(static_path):
            print(f"FAILED: CSS not found at {static_path}")
            return False
        
        print("SUCCESS: Flask app and templates configured correctly")
        return True
        
    except Exception as e:
        print(f"FAILED: Flask app error - {e}")
        return False

def test_config():
    """Test configuration."""
    print("Testing configuration...")
    try:
        from config.settings import Config
        
        # Test config validation
        config_status = Config.validate_config()
        print(f"Config status: {config_status}")
        
        # Check required locations
        if len(Config.VALID_LOCATIONS) != 10:
            print(f"WARNING: Expected 10 locations, found {len(Config.VALID_LOCATIONS)}")
        
        print("SUCCESS: Configuration loaded")
        assert True
        
    except Exception as e:
        print(f"FAILED: Config error - {e}")
        assert False, f"Config error: {e}"

def main():
    """Run all tests."""
    print("Indra Travel Bot - Application Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_flask_app,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("SUCCESS: All tests passed!")
        print("\nYou can now run the application:")
        print("  python app.py")
        return True
    else:
        print("FAILED: Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)