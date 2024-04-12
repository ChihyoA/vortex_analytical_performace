import pandas as pd
import numpy as np

# Specify the path to your CSV file
file_path = 'perf_counter_final.csv'
output_file_path = 'perf_comparison_output.csv'

# Set Testing Environment
# Cores, Warps, Threads, L2, Mapping
Biggest_configuration     = (16, 32, 4, 2, 'TM')
Most_common_configuration = (16,  2, 8, 2, 'TM')

Static_analysis_configuation = (16, 2, 8, 2, 'TM')

app_list = ['bfs',
 'blackscholes',
 'convolution',
 'dotproduct',
 'guassian',
 'kmeans',
 'lbm',
 'nearn',
 'psort',
 'saxpy',
 'sfilter',
 'sgemm',
 'spmv',
 'stencil',
 'transpose',
 'vecadd']
 

def find_lowest_cycles_app_data(data, app_name, max_ALU=256):
    """
    Finds the app data (index values) for the entry with the lowest 'cycles' for a given app name.
    
    Parameters:
    - data: A pandas DataFrame with a multi-level index set as ["APP name", "Cores", "Warps", "Threads", "L2", "Mapping"].
    - app_name: The name of the app to search for.
    
    Returns:
    - A tuple containing the index values ["APP name", "Cores", "Warps", "Threads", "L2", "Mapping"] of the entry with the lowest 'cycles' for the specified app name. If the app name is not found, returns None.
    """
    try:
        # Filter the DataFrame for the specified app name
        filtered_data = data.xs(app_name, level='APP name')
        non_zero_cycles_data = filtered_data[filtered_data['cycles'] != 0]
        non_zero_cycles_data_area_constraint = non_zero_cycles_data[
        (non_zero_cycles_data.index.get_level_values('Cores') * 
        non_zero_cycles_data.index.get_level_values('Threads') * 
        np.minimum(non_zero_cycles_data.index.get_level_values('Warps'), 4)) <= max_ALU]
        
        # Sort the filtered DataFrame by 'cycles' in ascending order
        sorted_data = non_zero_cycles_data_area_constraint.sort_values(by='cycles')
        
        # Get the index of the row with the lowest cycles
        if not sorted_data.empty:
            lowest_cycles_index = sorted_data.index[0]
            # Return the index as a tuple
            return lowest_cycles_index, sorted_data.loc[lowest_cycles_index].loc['cycles']
        else:
            print("No entry with non-zero cycles found for the specified app name.")
            return None
    except KeyError:
        print("There was a KeyError while trying to find the specified app name.", app_name)
        # In case the specified app name is not in the DataFrame
        return None
    
def find_cycles_at_given_config(data, app_name, config_tuple):

    configuration_row = data.loc[(app_name,) + config_tuple]
    configuration_cycles = configuration_row['cycles']
    configuration_instrs = configuration_row['instrs']
    
    if configuration_cycles == 0:
        if configuration_instrs == 0:
            return "Failed"
        else:
            return "Run took long"
    
    return configuration_cycles
    





# Use pandas to read the CSV file. Assuming the first row contains the headers.
data = pd.read_csv(file_path)

# Now, `data` is a DataFrame object containing all the CSV data.
data.set_index(["APP name", "Cores", "Warps", "Threads", "L2", "Mapping"], inplace=True)

data_to_export = []
for app in app_list:
    print("Processing app:", app)

    Best_congiruation, Best_cycles = find_lowest_cycles_app_data(data, app)
    Biggest_configuration_cycles = find_cycles_at_given_config(data, app, Biggest_configuration)
    Most_common_configuration_cycles = find_cycles_at_given_config(data, app, Most_common_configuration)
    Static_analysis_configuation_cycles = find_cycles_at_given_config(data, app, Static_analysis_configuation)

    try:
        Biggest_configuration_cycles_relative = int(Biggest_configuration_cycles)/int(Best_cycles)
    except:
        Biggest_configuration_cycles_relative = "N/A"
    try:
        Most_common_configuration_cycles_relative = int(Most_common_configuration_cycles)/int(Best_cycles)
    except:
        Most_common_configuration_cycles_relative = "N/A"
    try:
        Static_analysis_configuation_cycles_relative = int(Static_analysis_configuation_cycles)/int(Best_cycles)
    except:
        Static_analysis_configuation_cycles_relative = "N/A"



    data_to_export.append({
        "App": app,
        "Best configuration": Best_congiruation,
        "Best cycles": Best_cycles,
        "Biggest configuration": Biggest_configuration,
        "Biggest configuration cycles": Biggest_configuration_cycles,
        "Biggest configuration cycles(relative)": Biggest_configuration_cycles_relative,
        "Most common configuration": Most_common_configuration,
        "Most common configuration cycles": Most_common_configuration_cycles,
        "Most common configuration cycles(relative)": Most_common_configuration_cycles_relative,
        "Static analysis configuration": Static_analysis_configuation,
        "Static analysis configuration cycles": Static_analysis_configuation_cycles,
        "Static analysis configuration cycles(relative)": Static_analysis_configuation_cycles_relative
    })



# Now, you can perform lookups using .loc with a tuple representing the values of your keys
# For example, to look up data for a specific combination of keys:
df_to_export = pd.DataFrame(data_to_export)
df_to_export.to_csv(output_file_path, index=False)

# If you need to reset the index to its default integer-based index later, you can do so with:
data.reset_index(inplace=True)
