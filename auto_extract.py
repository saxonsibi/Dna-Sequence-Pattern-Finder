import os
import re
import ast
from pathlib import Path

# --- Heuristics for Discovery ---
# We now look for both .py and .ipynb files
POTENTIAL_MAIN_SCRIPTS = ['main.py', 'app.py', 'run.py', 'ncbi_dataset.ipynb', 'train_classifier.py', 'dna_flask.py']
README_PATTERNS = ['README.md', 'README.rst', 'README.txt']
STANDARD_LIBRARY = {
    'os', 'sys', 're', 'json', 'math', 'random', 'datetime', 'collections',
    'itertools', 'functools', 'operator', 'pathlib', 'string', 'csv', 'argparse'
}

# --- NEW: Helper function to get source code from .py or .ipynb ---
def get_source_code_from_file(script_path):
    """Reads and returns the source code from a .py or .ipynb file."""
    if script_path.endswith('.ipynb'):
        try:
            import nbformat
        except ImportError:
            print("❌ Error: 'nbformat' library is required to parse Jupyter Notebooks.")
            print("Please install it by running: pip install nbformat")
            return None
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            source_code = ""
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    # Join lines of code from each cell
                    source_code += "\n".join(cell.source) + "\n"
            return source_code
        except Exception as e:
            print(f"Error reading notebook {script_path}: {e}")
            return None
    else: # It's a .py file
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading script {script_path}: {e}")
            return None

def discover_project_structure():
    """Automatically finds key files in the project directory."""
    structure = {
        'main_script': None,
        'readme': None,
        'requirements': None,
        'dockerfile': None,
        'github_actions': False
    }

    # 1. Find README
    for pattern in README_PATTERNS:
        if Path(pattern).exists():
            structure['readme'] = pattern
            break
    
    # 2. Find requirements.txt
    if Path('requirements.txt').exists():
        structure['requirements'] = 'requirements.txt'

    # 3. Find Dockerfile
    if Path('Dockerfile').exists():
        structure['dockerfile'] = 'Dockerfile'

    # 4. Find GitHub Actions workflows
    if Path('.github/workflows').exists():
        structure['github_actions'] = True

    # 5. Find the main script (.py or .ipynb)
    for script_name in POTENTIAL_MAIN_SCRIPTS:
        if Path(script_name).exists():
            structure['main_script'] = script_name
            break
    
    # If not found, find the file with the most functions/code cells
    if not structure['main_script']:
        max_items = -1
        candidate_script = None
        for file_path in Path('.').glob('*.py'):
             if file_path.name.startswith('test_') or file_path.name == '__init__.py': continue
             source = get_source_code_from_file(file_path)
             if source:
                tree = ast.parse(source)
                item_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                if item_count > max_items:
                    max_items = item_count
                    candidate_script = file_path.name
        
        for file_path in Path('.').glob('*.ipynb'):
             source = get_source_code_from_file(file_path)
             if source:
                tree = ast.parse(source)
                item_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                if item_count > max_items:
                    max_items = item_count
                    candidate_script = file_path.name
        
        if candidate_script:
            structure['main_script'] = candidate_script
            
    return structure

def parse_imports_from_source(source_code):
    """Extracts imported library names from a string of source code."""
    if not source_code: return []
    imported_libs = set()
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_libs.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_libs.add(node.module.split('.')[0])
        return sorted(list(imported_libs - STANDARD_LIBRARY))
    except Exception:
        return []

def extract_readme_overview(readme_path):
    """Extracts the overview section from a given README file."""
    if not readme_path: return "A powerful tool for data analysis and automation."
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'## 🌟 Overview\s*\n(.*?)(?=\n##|\n#|\Z)', content, re.DOTALL)
        if match: return match.group(1).strip()
        match = re.search(r'^(.*?)(?=\n\n|\n#|\n##)', content, re.DOTALL)
        if match: return match.group(1).strip()
    except Exception: pass
    return "No project overview found."

def analyze_code_features_from_source(source_code):
    """Analyzes source code to find key function names."""
    if not source_code: return []
    features = []
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if any(keyword in node.name.lower() for keyword in ['find', 'search', 'calculate', 'analyze', 'parse', 'plot', 'run', 'main', 'fetch', 'clean', 'process']):
                    features.append(node.name)
        return features
    except Exception:
        return []

def format_features_for_linkedin(features):
    """Turns function names into impressive bullet points."""
    feature_map = {
        "find_gc_content": "Developed a function to calculate GC content, a key indicator of DNA stability.",
        "search_motif": "Implemented a flexible motif search using regular expressions to find specific DNA patterns.",
        "find_orfs": "Created an algorithm to identify potential Open Reading Frames (ORFs) in DNA sequences.",
        "plot_gc_content": "Generated data visualizations using Matplotlib to plot GC content across sequence regions.",
        "fetch_data": "Automated the retrieval of genomic datasets from online databases like NCBI.",
        "clean_data": "Engineered a data cleaning pipeline to handle missing values and format inconsistencies.",
        "main": "Engineered the main application logic and command-line interface.",
        "run_analysis": "Orchestrated the end-to-end data analysis pipeline."
    }
    
    formatted_points = []
    for feature in features[:5]:
        description = feature_map.get(feature, f"Implemented the `{feature}` module for core analysis.")
        formatted_points.append(f"- {description}")
    return "\n".join(formatted_points)

def generate_content():
    """Main function to discover and generate content."""
    print("🔍 Discovering project structure...")
    project = discover_project_structure()
    
    if not project['main_script']:
        print("❌ Could not find a main script (.py or .ipynb). Exiting.")
        return

    print(f"✅ Found main script: {project['main_script']}")
    print(f"✅ Found README: {project['readme'] or 'None'}")
    print(f"✅ Found dependencies: {project['requirements'] or 'Will parse imports'}")
    
    # 1. Get source code from the main file (py or ipynb)
    source_code = get_source_code_from_file(project['main_script'])
    if not source_code:
        print("❌ Could not read source code from the main script. Exiting.")
        return

    # 2. Extract Tech Stack and Features from the source code
    tech_stack = parse_imports_from_source(source_code)
    code_features = analyze_code_features_from_source(source_code)
    
    # 3. Extract other data
    overview = extract_readme_overview(project['readme'])
    
    # 4. Format for GitHub
    project_name = Path('.').resolve().name.replace('-', ' ').replace('_', ' ').title()
    github_desc = f"A {project_name} tool for bioinformatics and data analysis."
    
    # 5. Format for LinkedIn
    infrastructure_points = []
    if project['dockerfile']: infrastructure_points.append("- Containerized the application using Docker for easy deployment.")
    if project['github_actions']: infrastructure_points.append("- Implemented CI/CD pipelines with GitHub Actions for automated testing and integration.")

    linkedin_content = f"""### {project_name}

**Project Overview:**
{overview}

**Key Accomplishments & Features:**
I developed a Python-based tool to perform critical analyses on genomic data. Key contributions include:
{format_features_for_linkedin(code_features)}
{chr(10).join(infrastructure_points)}

**Technical Stack:**
Python, {', '.join(tech_stack) if tech_stack else 'Standard Libraries'}
"""
    
    # 6. Print the results
    print("\n" + "---" * 10)
    print(">>> CONTENT FOR GITHUB REPOSITORY DESCRIPTION <<<")
    print("---" * 10)
    print(github_desc)
    print("\n" + "---" * 10)
    print(">>> CONTENT FOR LINKEDIN PROJECT SECTION <<<")
    print("---" * 10)
    print(linkedin_content)


if __name__ == "__main__":
    generate_content()