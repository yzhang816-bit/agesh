print("Start")
import sys
print(sys.version)
try:
    import pandas
    print("Pandas imported")
except ImportError as e:
    print(f"Pandas failed: {e}")
print("End")
