"""
Author: Cameron Murphy (Student ID: 104967811, GitHub: 0x104967811)
Date: June 5th 2025
"""
from typing import Any
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import subprocess
import sys
import time
import threading
import warnings
import shutil

# Suppress warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    os.environ["PYTHONWARNINGS"] = "ignore"

# Terminal width for centering
TERMINAL_WIDTH = shutil.get_terminal_size((80, 20)).columns

def print_centered(text, width=TERMINAL_WIDTH):
    """Print text centered in terminal."""
    print(text.center(width))

def create_rainbow_title():
    """Create title with colours."""
    # colours (ANSI escape codes)
    colours = [
        '\033[38;2;255;182;193m',  # Light Pink
        '\033[38;2;255;218;185m',  # Peach
        '\033[38;2;255;255;224m',  # Light Yellow
        '\033[38;2;193;255;193m',  # Light Green
        '\033[38;2;173;216;230m',  # Light Blue
        '\033[38;2;221;160;221m',  # Plum
        '\033[38;2;230;230;250m',  # Lavender
    ]
    reset = '\033[0m'

    # Create the title
    print("\n" * 2)

    # Top border with gradient
    border_top = "╔" + "═" * 58 + "╗"
    gradient_border_top = ""
    for i, char in enumerate(border_top):
        gradient_border_top += colours[i % len(colours)] + char
    print_centered(gradient_border_top + reset)

    # Title lines
    title_lines = [
        "║                                                          ║",
        "║         ██╗███╗   ██╗██████╗ ██████╗  █████╗             ║",
        "║         ██║████╗  ██║██╔══██╗██╔══██╗██╔══██╗            ║",
        "║         ██║██╔██╗ ██║██║  ██║██████╔╝███████║            ║",
        "║         ██║██║╚██╗██║██║  ██║██╔══██╗██╔══██║            ║",
        "║         ██║██║ ╚████║██████╔╝██║  ██║██║  ██║            ║",
        "║         ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝            ║",
        "║                   TRAVEL CHATBOT                         ║",
        "║                                                          ║",
    ]

    for line_num, line in enumerate(title_lines):
        coloured_line = ""
        for i, char in enumerate(line):
            if char in "█╗╔╝╚═║INDRA":
                # Apply extra effect to special characters
                colour_index = (i + line_num) % len(colours)
                coloured_line += colours[colour_index] + char + reset
            else:
                coloured_line += char
        print_centered(coloured_line)

    # Bottom border with gradient
    border_bottom = "╚" + "═" * 58 + "╝"
    gradient_border_bottom = ""
    for i, char in enumerate(border_bottom):
        gradient_border_bottom += colours[i % len(colours)] + char
    print_centered(gradient_border_bottom + reset)

    print("\n")

class LoadingBar:
    """Animated loading bar for package installation."""

    def __init__(self, total_steps=100):
        self.total_steps = total_steps
        self.current_step = 0
        self.running = False
        self.thread = None
        self.status_text = ""

    def start(self, status_text="Installing"):
        """Start the loading animation."""
        self.running = True
        self.status_text = status_text
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()

    def _animate(self):
        """Animate the loading bar."""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        colours = ['\033[95m', '\033[94m', '\033[96m', '\033[92m', '\033[93m']

        i = 0
        while self.running:
            # Create progress bar
            progress = int((self.current_step / self.total_steps) * 30)
            bar = '█' * progress + '░' * (30 - progress)

            # Colour selection
            colour = colours[i % len(colours)]
            reset = '\033[0m'

            # Display
            sys.stdout.write(f'\r  {colour}{spinner[i % len(spinner)]}{reset} {self.status_text:<30} [{colour}{bar}{reset}] {self.current_step}%')
            sys.stdout.flush()

            i += 1
            time.sleep(0.1)

    def update(self, step, status_text=None):
        """Update progress."""
        self.current_step = min(step, self.total_steps)
        if status_text:
            self.status_text = status_text

    def stop(self, success=True):
        """Stop the loading animation."""
        self.running = False
        if self.thread:
            self.thread.join()

        # Final status
        if success:
            status = f'  \033[92m✓\033[0m {self.status_text:<30} [{"█" * 30}] Complete!'
        else:
            status = f'  \033[91m✗\033[0m {self.status_text:<30} Failed'

        sys.stdout.write(f'\r{status}\n')
        sys.stdout.flush()

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

def parse_requirements():
    """Parse requirements.txt and separate PyPI packages from Git URLs."""
    pypi_requirements: list[Any] = []
    git_requirements: list[Any] = []

    requirements_file = os.path.join(this_directory, 'requirements.txt')

    if not os.path.exists(requirements_file):
        print("Warning: requirements.txt not found")
        return pypi_requirements, git_requirements

    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Separate git URLs from regular packages
            if line.startswith('git+') or 'git+' in line:
                git_requirements.append(line)
            else:
                pypi_requirements.append(line)

    return pypi_requirements, git_requirements

class Install(install):
    """installation with warnings suppressed."""

    def run(self):
        """Run installation with all warnings suppressed."""
        original_filters = warnings.filters[:]
        warnings.filterwarnings("ignore")

        # Suppress setuptools specific warnings
        if hasattr(self, '_warn_deprecated'):
            self._warn_deprecated = lambda: None

        try:
            # Display title
            create_rainbow_title()

            print_centered("Starting Indra Travel Bot Installation")
            print_centered("━" * 60)
            print()

            # Parse requirements
            _, git_requirements = parse_requirements()

            # Install Git dependencies with loading bar because it looks cool
            if git_requirements:
                loader = LoadingBar()
                loader.start("Installing ChatterBot fork")

                for i, git_req in enumerate(git_requirements):
                    try:
                        # Progress update
                        loader.update(int((i + 1) / len(git_requirements) * 50),
                                    "Installing ChatterBot fork")

                        # Install silently
                        subprocess.check_call([
                            sys.executable, '-m', 'pip', 'install',
                            '--quiet', '--disable-pip-version-check', git_req
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    except subprocess.CalledProcessError:
                        loader.stop(success=False)
                        print("     Please install manually: pip install " + git_req)

                loader.stop(success=True)

            # Install core dependencies with loading bar
            loader = LoadingBar()
            loader.start("Installing core dependencies")

            # Simulate progress during parent installation
            for i in range(0, 60, 10):
                loader.update(50 + i//2, "Installing core dependencies")
                time.sleep(0.2)

            # Run parent install silently
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    super().run()
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            loader.update(75, "Installing core dependencies")
            loader.stop(success=True)

            # Install spaCy model
            self.install_spacy_model()

            # Post-installation tasks
            self.post_install()

        finally:
            # Restore warning filters
            warnings.filters[:] = original_filters

    def install_spacy_model(self):
        """Install spaCy English model required for ChatterBot."""
        loader = LoadingBar()
        loader.start("Installing language model")

        try:
            for i in range(0, 80, 20):
                loader.update(75 + i//4, "Installing language model")
                time.sleep(0.1)

            subprocess.check_call([
                sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            loader.update(100, "Installing language model")
            loader.stop(success=True)

        except subprocess.CalledProcessError:
            loader.stop(success=False)
            print("     Run manually: python -m spacy download en_core_web_sm")

    def post_install(self):
        """Post-installation setup tasks."""
        loader = LoadingBar()
        loader.start("Setting up project structure")

        # Create necessary directories
        directories = [
            'data/databases',
            'data/cache',
            'data/training',
            'logs'
        ]

        for i, directory in enumerate(directories):
            os.makedirs(directory, exist_ok=True)
            loader.update(int((i + 1) / len(directories) * 100),
                         f"Creating {directory}")
            time.sleep(0.1)

        loader.stop(success=True)

        # Check for environment configuration
        print("\n  Checking environment configuration...", end="")
        if not os.path.exists('.env'):
            print(" ! Not found")
            print("\n  \033[93mIMPORTANT:\033[0m Create a .env file with:")
            print("    • OPENWEATHER_API_KEY=your_key_here")
            print("    • NEWS_API_KEY=your_key_here")
        else:
            print(" ✓")

        # Success message
        print()
        print_centered("━" * 60)
        print_centered("\033[92m Installation Complete! \033[0m")
        print_centered("━" * 60)
        print()
        print_centered("Next steps:")
        print_centered("1. Configure API keys in .env file")
        print_centered("2. Run 'python app.py' to start")
        print_centered("3. Visit http://localhost:5000")
        print()
        print_centered("\033[95m Ready to chat with Indra! \033[0m")
        print_centered("━" * 60)
        print()

class CustomDevelopCommand(develop):
    """Custom development command"""

    def run(self):
        """Run the custom development installation."""
        custom_install = Install(self.distribution)
        custom_install.run()
        develop.run(self)

# Parse requirements
pypi_requirements, git_requirements = parse_requirements()

# Default to install if no arguments
if __name__ == "__main__" and len(sys.argv) == 1:
    sys.argv.append('install')

setup(
    name="indra-chat-bot",
    version="1.0.0",
    author="Cameron Murphy",
    author_email="104967811@student.swin.edu.au",
    description="Flask-based chatbot for weather forecasts and activity recommendations for a few locations in England.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/0x104967811/Indra_Go_Travel_Bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: Flask",
    ],
    python_requires=">=3.8",
    install_requires=pypi_requirements,
    cmdclass={
        'install': Install,
        'develop': CustomDevelopCommand,
    },
    entry_points={
        'console_scripts': [
            'indra-bot=app:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="chatbot travel weather england flask chatterbot ai nlp",
    project_urls={
        "Source": "https://github.com/0x1049678II/Indra_Go_Travel_Bot",
        "Documentation": "https://github.com/0x1049678II/Indra_Go_Travel_Bot/readme.md",
    },
)
