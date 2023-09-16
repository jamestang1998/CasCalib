import csv

def csv_to_dict(filename):
    result_dict = {}
    
    with open(filename, mode='r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if len(row) >= 5:
                key = '_'.join(row[:4])
                value = row[4]
                result_dict[key] = value
    
    return result_dict

# Replace 'your_input.csv' with the path to your CSV file
csv_filename = '/local/tangytob/Summer2023/multiview_synchronization/scripts_multicam/sync_results/time_20230816_164448_pred_focal/result_average_sync.csv'
output_dict = csv_to_dict(csv_filename)

#for key, value in output_dict.items():
#    print(f'{key}: {value}')

for key, value in output_dict.items():
    print(f'"{key}": {value},')