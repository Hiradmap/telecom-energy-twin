import os
import sys

# Make `import tests...` and `import src...` work however pytest is invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
