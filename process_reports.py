import json
import sys


if __name__ == '__main__':
    
    report_files = ['simulation_report_baseline.json', 'simulation_report_divisive.json', 'simulation_report_bottom_up.json']

    for report_file in report_files:
        report = json.load(open(report_file, 'rb'))
        print(f'Report file: {report_file}')
        print(f'Total: {report["total"]}')
    
        num_nodes_recv = 0
        max_timestamp = 0
        min_timestamp = sys.maxsize
        for timestamp in report["log"]["emitted"]["message 0"].keys():
            num_nodes = report["log"]["emitted"]["message 0"][timestamp]
            num_nodes_recv += num_nodes
            max_timestamp = max(max_timestamp, int(timestamp))
            min_timestamp = min(min_timestamp, int(timestamp))

        print(f'{num_nodes_recv} nodes received the message in {max_timestamp} ms')
        print()
        print()

