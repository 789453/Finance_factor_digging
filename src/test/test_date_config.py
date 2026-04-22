import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from alphagen_generic.config import get_date_range

def test_date_config():
    print("Testing date configuration...")
    
    test_cases = [
        {"domain": "A", "train_end": 2021, "test_end": 2025},
        {"domain": "C", "train_end": 2021, "test_end": 2025},
        {"domain": "B", "train_end": 2023, "test_end": 2025},
    ]
    
    for tc in test_cases:
        train_start, train_end, test_start, test_end = get_date_range(
            tc["domain"], tc["train_end"], tc["test_end"]
        )
        print(f"\nDomain: {tc['domain']}, Train End Year: {tc['train_end']}, Test End Year: {tc['test_end']}")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Test:  {test_start} to {test_end}")
        
        # Verify continuity
        # test_start should be Jan 1st of the year after train_end
        expected_test_start = f"{tc['train_end'] + 1}0101"
        assert test_start == expected_test_start, f"Discontinuity detected! Expected {expected_test_start}, got {test_start}"
        
        # Verify domain start logic
        if tc["domain"] == "C":
            assert train_start == "20180101", f"Expected Domain C start 20180101, got {train_start}"
        else:
            assert train_start == "20100101", f"Expected default start 20100101, got {train_start}"
            
    print("\nAll date configuration tests passed! Continuity is guaranteed.")

if __name__ == "__main__":
    test_date_config()
