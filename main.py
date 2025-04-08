import os
import sys

# Ensure the scripts directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))

# Import the refactored script
import refactored_drowsy_detection

def main():
    print("Starting Drowsy Driver Detection Pipeline...")
    refactored_drowsy_detection.run()  # Call the main function in the refactored script

if __name__ == "__main__":
    main()
