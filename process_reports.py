import json
import sys
import numpy as np


if __name__ == '__main__':
    report_files = ['simulation_report_baseline.json', 'simulation_report_divisive.json', 'simulation_report_ring.json', 'simulation_report_bottom_up.json']
    
    summary_stats = {}
    for report_file in report_files:
        report = json.load(open(report_file, 'rb'))

        print(report)
        print()
        
        print(f'Report file: {report_file}')
        print(f'Total: {report["total"]}')

        print(f'{report["log"]["emitted"]}')
        for message in report["log"]["emitted"]:
            max_timestamp = 0
            min_timestamp = sys.maxsize
            for timestamp in report['log']['emitted'][message].keys():
                max_timestamp = max(max_timestamp, int(timestamp))
                min_timestamp = min(min_timestamp, int(timestamp))
            if report_file not in summary_stats:
                summary_stats[report_file] = {}
            if 'broadcast_duration' not in summary_stats[report_file]:
                summary_stats[report_file]['broadcast_duration'] = []
            summary_stats[report_file]['broadcast_duration'].append(max_timestamp - min_timestamp)

        print()
        print()
    
    print("SUMMARY:")
    for report_file in summary_stats:
        print(f'Report file: {report_file}')
        for metric in summary_stats[report_file].keys():
            print(f'{metric}: {np.mean(summary_stats[report_file][metric])} ±{np.std(summary_stats[report_file][metric])}')
        print()



