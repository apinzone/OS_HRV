#Identifies R-R intervals in the NeuroKit2 array that are above the max or below the min R-R distance from the ChronOS R-R interval array to further investigate discrepancies in peak detection
import pandas as pd
import numpy as np

def find_outlier_intervals(excel_file="validation_results.xlsx"):
    """
    Find NeuroKit2 R-R intervals that fall outside ChronOS min/max range
    """
    
    # Load both datasets
    chronos_df = pd.read_excel(excel_file, sheet_name='ChronOS_Results')
    neurokit_df = pd.read_excel(excel_file, sheet_name='NeuroKit2_Results')
    
    # Files with known peak discrepancies
    discrepancy_files = [
        'synthetic_ecg_015.edf', 'synthetic_ecg_041.edf', 'synthetic_ecg_053.edf',
        'synthetic_ecg_055.edf', 'synthetic_ecg_068.edf', 'synthetic_ecg_073.edf',
        'synthetic_ecg_078.edf', 'synthetic_ecg_082.edf', 'synthetic_ecg_092.edf',
        'synthetic_ecg_095.edf'
    ]
    
    print("R-R INTERVAL OUTLIER ANALYSIS")
    print("="*40)
    
    for filename in discrepancy_files:
        print(f"\n{filename}:")
        
        # Get data for this file
        chronos_row = chronos_df[chronos_df['filename'] == filename]
        neurokit_row = neurokit_df[neurokit_df['filename'] == filename]
        
        if chronos_row.empty or neurokit_row.empty:
            print(f"  ERROR: File not found")
            continue
            
        # Extract R-R interval arrays
        chronos_rr = eval(chronos_row['rr_intervals_ms'].iloc[0])
        neurokit_rr = eval(neurokit_row['rr_intervals_ms'].iloc[0])
        
        # Calculate ChronOS range
        chronos_min = min(chronos_rr)
        chronos_max = max(chronos_rr)
        
        print(f"  ChronOS range: {chronos_min:.1f} - {chronos_max:.1f} ms")
        print(f"  ChronOS intervals: {len(chronos_rr)}")
        print(f"  NeuroKit2 intervals: {len(neurokit_rr)}")
        
        # Get peak timestamps for mapping intervals to times
        neurokit_peaks = eval(neurokit_row['r_peak_timestamps_sec'].iloc[0])
        
        # Find NeuroKit2 intervals outside ChronOS range with timestamps
        outliers_below = []
        outliers_above = []
        
        for i, interval in enumerate(neurokit_rr):
            if interval < chronos_min:
                timestamp = neurokit_peaks[i]  # Start time of this interval
                outliers_below.append((interval, timestamp, i))
            elif interval > chronos_max:
                timestamp = neurokit_peaks[i]  # Start time of this interval
                outliers_above.append((interval, timestamp, i))
        
        if outliers_below:
            print(f"  NeuroKit2 intervals BELOW ChronOS min ({chronos_min:.1f} ms):")
            for interval, timestamp, idx in outliers_below[:5]:  # Show first 5
                print(f"    {interval:.1f} ms at {timestamp:.2f}s (interval #{idx})")
            if len(outliers_below) > 5:
                print(f"    ... and {len(outliers_below) - 5} more")
            print(f"    Total: {len(outliers_below)} intervals")
            
        if outliers_above:
            print(f"  NeuroKit2 intervals ABOVE ChronOS max ({chronos_max:.1f} ms):")
            for interval, timestamp, idx in outliers_above[:5]:  # Show first 5
                print(f"    {interval:.1f} ms at {timestamp:.2f}s (interval #{idx})")
            if len(outliers_above) > 5:
                print(f"    ... and {len(outliers_above) - 5} more")
            print(f"    Total: {len(outliers_above)} intervals")
            
        if not outliers_below and not outliers_above:
            print(f"  All NeuroKit2 intervals within ChronOS range")
             
        neurokit_min = min(neurokit_rr)
        neurokit_max = max(neurokit_rr)
        print(f"  NeuroKit2 range: {neurokit_min:.1f} - {neurokit_max:.1f} ms")

def main():
    find_outlier_intervals()

if __name__ == "__main__":
    main()