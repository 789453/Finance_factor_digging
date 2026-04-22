import os
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    print("pyarrow not installed, please install it to test parquet files")
    sys.exit(1)

def check_parquet_files(directory):
    total_files = 0
    valid_files = 0
    invalid_files = []
    
    print(f"Scanning {directory} for parquet files...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet'):
                total_files += 1
                filepath = os.path.join(root, file)
                try:
                    # Attempt to read parquet metadata
                    pq.ParquetFile(filepath)
                    valid_files += 1
                except Exception as e:
                    invalid_files.append((filepath, str(e)))
                    
    print(f"\n--- Summary ---")
    print(f"Total parquet files found: {total_files}")
    print(f"Valid parquet files: {valid_files}")
    if invalid_files:
        print(f"Invalid parquet files ({len(invalid_files)}):")
        for f, err in invalid_files:
            print(f"  - {f}: {err}")
    else:
        print("All parquet files are intact.")

if __name__ == "__main__":
    check_parquet_files('/mnt')
