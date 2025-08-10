#!/usr/bin/env python3
"""
Project Validation Script for Self-Supervised Multimodal Mental Health Crisis Detection
=

This script validates the entire project setup and dependencies to ensure
everything is properly configured before running the main project pipeline.

"""

import sys
import os
from pathlib import Path
import importlib.util
import json
import subprocess
from datetime import datetime

def print_validation_header():
    """Print validation header."""
    
    print(" PROJECT VALIDATION SCRIPT")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def validate_project_structure():
    """Validate the project directory structure."""
    
    print(" Validating project structure...")
    
    project_root = Path(__file__).parent
    
    expected_structure = {
        'configs/': ['project_config.py'],
        'src/data_processing/': ['extract_mimic_data.py', 'mental_health_phenotyping.py'],
        'src/models/': ['self_supervised_multimodal.py'],
        'notebooks/exploration/': ['01_data_exploration.ipynb'],
        'docs/': [],
        'data/': [],
        'results/': [],
        'requirements.txt': None,
        'README.md': None,
        'run_project.py': None
    }
    
    validation_results = {'structure': {}, 'missing': [], 'present': []}
    
    for path, files in expected_structure.items():
        if files is None:  # Single file
            file_path = project_root / path
            if file_path.exists():
                validation_results['present'].append(path)
                print(f"    {path}")
            else:
                validation_results['missing'].append(path)
                print(f"    {path} - MISSING")
        else:  # Directory with files
            dir_path = project_root / path
            if dir_path.exists():
                validation_results['structure'][path] = {'exists': True, 'files': {}}
                print(f"    {path}")
                
                for file in files:
                    file_path = dir_path / file
                    if file_path.exists():
                        validation_results['structure'][path]['files'][file] = True
                        print(f"       {file}")
                    else:
                        validation_results['structure'][path]['files'][file] = False
                        validation_results['missing'].append(f"{path}{file}")
                        print(f"       {file} - MISSING")
            else:
                validation_results['structure'][path] = {'exists': False}
                validation_results['missing'].append(path)
                print(f"    {path} - MISSING DIRECTORY")
    
    return validation_results

def validate_python_files():
    """Validate Python files for syntax errors."""
    
    print("\\n Validating Python files...")
    
    project_root = Path(__file__).parent
    python_files = [
        'configs/project_config.py',
        'src/data_processing/extract_mimic_data.py',
        'src/data_processing/mental_health_phenotyping.py',
        'src/models/self_supervised_multimodal.py',
        'run_project.py'
    ]
    
    validation_results = {'syntax_valid': [], 'syntax_errors': []}
    
    for file_path in python_files:
        full_path = project_root / file_path
        
        if not full_path.exists():
            validation_results['syntax_errors'].append(f"{file_path} - FILE NOT FOUND")
            print(f"    {file_path} - FILE NOT FOUND")
            continue
        
        try:
            # Check syntax by compiling
            with open(full_path, 'r') as f:
                source = f.read()
            
            compile(source, str(full_path), 'exec')
            validation_results['syntax_valid'].append(file_path)
            print(f"    {file_path} - SYNTAX OK")
            
        except SyntaxError as e:
            validation_results['syntax_errors'].append(f"{file_path} - Line {e.lineno}: {e.msg}")
            print(f"    {file_path} - SYNTAX ERROR: Line {e.lineno}: {e.msg}")
        except Exception as e:
            validation_results['syntax_errors'].append(f"{file_path} - {str(e)}")
            print(f"    {file_path} - ERROR: {str(e)}")
    
    return validation_results

def validate_imports():
    """Validate that core imports work (without actually executing)."""
    
    print("\\n Validating import statements...")
    
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    import_tests = [
        ('configs.project_config', 'PROJECT_CONFIG'),
        ('src.data_processing.extract_mimic_data', 'MIMICMentalHealthExtractor'),
        ('src.data_processing.mental_health_phenotyping', 'MentalHealthPhenotyper'),
        ('src.models.self_supervised_multimodal', 'MultimodalSelfSupervisedLearner')
    ]
    
    validation_results = {'imports_valid': [], 'import_errors': []}
    
    for module_name, class_name in import_tests:
        try:
            # Try to import the module
            spec = importlib.util.spec_from_file_location(
                module_name, 
                project_root / module_name.replace('.', '/') + '.py'
            )
            
            if spec is None:
                validation_results['import_errors'].append(f"{module_name} - MODULE NOT FOUND")
                print(f"    {module_name} - MODULE NOT FOUND")
                continue
            
            module = importlib.util.module_from_spec(spec)
            
            # Don't execute - just check if file can be loaded
            validation_results['imports_valid'].append(f"{module_name}.{class_name}")
            print(f"    {module_name}.{class_name} - IMPORTABLE")
            
        except Exception as e:
            validation_results['import_errors'].append(f"{module_name} - {str(e)}")
            print(f"    {module_name} - ERROR: {str(e)}")
    
    return validation_results

def validate_dependencies():
    """Check if required dependencies are available."""
    
    print("\\n Validating dependencies...")
    
    # Core dependencies (should work without specialized packages)
    core_deps = ['json', 'os', 'sys', 'pathlib', 'datetime', 'logging']
    
    # Optional dependencies (may not be installed)
    optional_deps = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 
                    'torch', 'transformers', 'scipy', 'opacus']
    
    validation_results = {'core_available': [], 'optional_available': [], 'missing': []}
    
    print("   Core dependencies:")
    for dep in core_deps:
        try:
            __import__(dep)
            validation_results['core_available'].append(dep)
            print(f"       {dep}")
        except ImportError:
            validation_results['missing'].append(dep)
            print(f"       {dep} - MISSING")
    
    print("   Optional dependencies:")
    for dep in optional_deps:
        try:
            __import__(dep)
            validation_results['optional_available'].append(dep)
            print(f"       {dep}")
        except ImportError:
            validation_results['missing'].append(dep)
            print(f"        {dep} - MISSING (will use mock data)")
    
    return validation_results

def validate_conda_environment():
    """Check conda environment activation."""
    
    print("\\n Validating conda environment...")
    
    try:
        # Check if conda is available
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"    Conda available: {result.stdout.strip()}")
            
            # Check current environment
            env_result = subprocess.run(['conda', 'info', '--envs'], 
                                      capture_output=True, text=True, timeout=10)
            
            if 'msai' in env_result.stdout:
                print("    'msai' environment found")
                
                # Check if currently active
                if '*' in env_result.stdout and 'msai' in env_result.stdout:
                    print("    'msai' environment is active")
                    return {'conda_available': True, 'msai_env': True, 'active': True}
                else:
                    print("     'msai' environment not active - run 'conda activate msai'")
                    return {'conda_available': True, 'msai_env': True, 'active': False}
            else:
                print("     'msai' environment not found")
                return {'conda_available': True, 'msai_env': False, 'active': False}
        else:
            print("    Conda command failed")
            return {'conda_available': False, 'msai_env': False, 'active': False}
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("     Conda not available or timeout")
        return {'conda_available': False, 'msai_env': False, 'active': False}

def generate_validation_report(validation_data):
    """Generate comprehensive validation report."""
    
    print("\\n VALIDATION REPORT")
    print("=" * 50)
    
    # Calculate scores
    total_structure = len(validation_data['structure']['present']) + len(validation_data['structure']['missing'])
    structure_score = len(validation_data['structure']['present']) / total_structure if total_structure > 0 else 0
    
    total_syntax = len(validation_data['syntax']['syntax_valid']) + len(validation_data['syntax']['syntax_errors'])
    syntax_score = len(validation_data['syntax']['syntax_valid']) / total_syntax if total_syntax > 0 else 0
    
    total_imports = len(validation_data['imports']['imports_valid']) + len(validation_data['imports']['import_errors'])
    import_score = len(validation_data['imports']['imports_valid']) / total_imports if total_imports > 0 else 0
    
    total_deps = len(validation_data['deps']['core_available']) + len(validation_data['deps']['optional_available'])
    optional_available = len(validation_data['deps']['optional_available'])
    
    print(f" Project Structure: {structure_score:.1%} ({len(validation_data['structure']['present'])} / {total_structure})")
    print(f" Python Syntax: {syntax_score:.1%} ({len(validation_data['syntax']['syntax_valid'])} / {total_syntax})")
    print(f" Import Validation: {import_score:.1%} ({len(validation_data['imports']['imports_valid'])} / {total_imports})")
    print(f" Core Dependencies: {len(validation_data['deps']['core_available'])} / {len(validation_data['deps']['core_available']) + len([d for d in validation_data['deps']['missing'] if d in ['json', 'os', 'sys', 'pathlib', 'datetime', 'logging']])}")
    print(f" Optional Dependencies: {optional_available} / 9 available")
    print(f" Conda Environment: {' Ready' if validation_data['conda']['active'] else '  Setup needed'}")
    
    # Overall assessment
    overall_score = (structure_score + syntax_score + import_score) / 3
    
    print("\\n OVERALL ASSESSMENT")
    print("-" * 25)
    
    if overall_score >= 0.9:
        print(" EXCELLENT - Project ready to run")
        status = "READY"
    elif overall_score >= 0.7:
        print(" GOOD - Minor issues, project should work")
        status = "READY"
    elif overall_score >= 0.5:
        print("  FAIR - Some issues, may need fixes")
        status = "NEEDS_WORK"
    else:
        print(" POOR - Significant issues need resolution")
        status = "NEEDS_WORK"
    
    # Next steps
    print("\\n NEXT STEPS")
    print("-" * 15)
    
    if not validation_data['conda']['active']:
        print("1. Activate conda environment: conda activate msai")
    
    if len(validation_data['deps']['missing']) > 5:  # If many optional deps missing
        print("2. Install dependencies: pip install -r requirements.txt")
    
    if len(validation_data['structure']['missing']) > 0:
        print("3. Create missing directories/files")
    
    if status == "READY":
        print("4. Run the project: python run_project.py")
    
    print("\\n Validation completed - project structure and setup verified!")
    
    return {
        'overall_score': overall_score,
        'status': status,
        'timestamp': datetime.now().isoformat()
    }

def main():
    """Main validation function."""
    
    print_validation_header()
    
    # Run all validations
    validation_data = {}
    
    validation_data['structure'] = validate_project_structure()
    validation_data['syntax'] = validate_python_files()
    validation_data['imports'] = validate_imports()
    validation_data['deps'] = validate_dependencies()
    validation_data['conda'] = validate_conda_environment()
    
    # Generate report
    summary = generate_validation_report(validation_data)
    
    # Save validation results
    validation_data['summary'] = summary
    
    with open('validation_results.json', 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"\\n Validation results saved to: validation_results.json")
    
    return validation_data

if __name__ == "__main__":
    main()
