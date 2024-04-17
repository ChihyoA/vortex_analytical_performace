import pandas as pd
import numpy as np

# Specify the path to your CSV file
file_path = 'perf_counter_final.csv'
output_file_path = 'perf_comparison_output.csv'

# Set Testing Environment
# Cores, Warps, Threads, L2, Mapping
Area_greedy_configuration           = (16, 16, 4, 2, 'CM')
Most_common_configuration           = (16,  2, 8, 2, 'CM')

Throughput_greedy_configuration     = (16,  2, 8, 2, 'CM') # placeholder
Occupancy_greedy_configuration      = (16,  2, 8, 2, 'CM') # placeholder

Static_analysis_configuration = (16, 2, 8, 2, 'TM')

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

def get_input_data(data, app_name, data_characteristic):
    """
    This function takes an app name as input and returns the workgroup size and number of workgroups
    for that app from the DataFrame.
    """
    # Filter the DataFrame for the given app_name
    filtered_data = data[data['app_name'] == app_name]
    
    # Return the workgroup size and number of workgroups if the app is found
    if not filtered_data.empty:
        return filtered_data[data_characteristic]
    else:
        return "Application not found."
    
def sort_and_filter_data(data, sort_order, filter_number, final_decision):
    for i in range(len(sort_order)):
        top_n_values = pd.Series(data[sort_order[i]].unique())
        data = data[data[sort_order[i]].isin(top_n_values.nlargest(filter_number[i]))]
        
    data = data.sort_values(by=final_decision, ascending=False)
    final_configuration = (data['C'].values[0], data['W'].values[0], data['T'].values[0], data['L2'].values[0], data['Mapping'].values[0])

    return final_configuration

def calculate_occupancy_and_depth(data, workgroup_size, num_workgroup, flop_ratio, mem_ratio):  
    ###########################Add User_defined info############################
    avg_flop_latency = 6
    avg_mem_latency = 20
    ############################################################################

    # Filtering the DataFrame to include only rows where ALUs are 256 or less
    area_confined = data[data['ALUs'] <= 256]

    # Initialize columns
    area_confined['occupancy_width'] = np.nan
    area_confined['occupancy_depth'] = np.nan

    # Apply calculations based on Mapping
    for index, row in area_confined.iterrows():
        if row['Mapping'] == 'TM':
            NT = num_workgroup
            
            area_confined.at[index, 'occupancy_width'] = (NT / row['W']) / np.ceil(NT / row['W'])
            area_confined.at[index, 'occupancy_depth'] = (np.ceil(NT / row['T']) / (row['W'] * row['C'])) / \
                                                np.ceil((np.ceil(NT / row['T']) / (row['W'] * row['C'])))
        elif row['Mapping'] == 'CM':
            # For CM, assuming #NB = W as a placeholder
            NT = workgroup_size
            NB = num_workgroup
            area_confined.at[index, 'occupancy_width'] = (NT / row['W']) / np.ceil(NT / row['W'])
            area_confined.at[index, 'occupancy_depth'] = (NB / row['C'] / np.ceil(NB / row['C'])) * \
                                                (np.ceil(NT / row['T']) / row['W']) / \
                                                np.ceil(np.ceil( NT / row['T']) / row['W'])
            
        area_confined.at[index, 'total_occupancy'] = area_confined.at[index, 'occupancy_width'] * area_confined.at[index, 'occupancy_depth']

        # Desired warp depth for Throughput greedy configuration
        area_confined.at[index, 'greedy_thruput_desired_depth'] = min(4, int(row['W'])) * (avg_flop_latency * flop_ratio + avg_mem_latency * mem_ratio)
    
    return area_confined

def apply_occupancy_greedy(analysis_group_index, analysis_index, data, app_name,
                            sort_order=['Registers', 'L1(Dcache)', 'L2'],
                            filter_number=[4, 2, 1],
                            final_decision='total_occupancy'):
    data = data[data['total_occupancy'] > 0.8]

    # export the sorted area_confied to csv
    data.to_csv('tmp/'+app_name+'_'+analysis_group_index+'_'+analysis_index+'_occupancy_greedy.csv', index=False)
    Occupancy_greedy_configuration = sort_and_filter_data(data, sort_order, filter_number, final_decision)  
    return data, Occupancy_greedy_configuration


def apply_throughput_greedy(analysis_group_index, analysis_index, data, app_name,
                            sort_order=['C', 'T', 'Registers'],
                            filter_number=[4, 2, 1],
                            final_decision='Registers'):
    ###########################Add User_defined info############################
    max_warp_width = 8    
    ############################################################################
    
    # Calculate the absolute difference with each row's depth
    data['temp_diff'] = abs(data['greedy_thruput_desired_depth'] - data['W'])

    # Find the row with the minimum difference for this depth
    data = data[data['temp_diff'] == data['temp_diff'].min()]
    
    data.to_csv('tmp/'+app_name+'_'+analysis_group_index+'_'+analysis_index+'_throughput_greedy.csv', index=False)

    data = data[data['ALUs'] == 256]

    Throughput_greedy_configuration = sort_and_filter_data(data, sort_order, filter_number, final_decision)  
    desired_warp_depth_from_throughput_greedy = [min(Throughput_greedy_configuration[1], max_warp_width)]

    return data, Throughput_greedy_configuration, desired_warp_depth_from_throughput_greedy

def apply_divergent_branch_analysis(analysis_group_index, analysis_index, data, app_name, divergent_branches,
                                    sort_order=['C', 'W', 'Registers'],
                                    filter_number=[4, 2, 1],
                                    final_decision='Registers'):
    
    # Divergent branch analysis for warp_width decision
    if divergent_branches == 0:
        data = data[data['T'] >= 16]
        desired_warp_width_from_linear_access = [16, 32]

    elif divergent_branches >= 1 and divergent_branches < 5:
        data = data[data['T'].isin([4, 8])]
        desired_warp_width_from_linear_access = [4, 8]

    else:
        data = data[data['T'] == 2]
        desired_warp_width_from_linear_access = [2]

    data.to_csv('tmp/'+app_name+'_'+analysis_group_index+'_'+analysis_index+'_divergent_branch.csv', index=False)

    Static_and_linear_branch_configuration = sort_and_filter_data(data, sort_order, filter_number, final_decision)



    return data, Static_and_linear_branch_configuration, desired_warp_width_from_linear_access

def apply_linear_access_analysis(analysis_group_index, analysis_index, data, app_name, num_load, num_store, num_linear_access,
                                sort_order=['C', 'T', 'Registers'],
                                filter_number=[4, 2, 1],
                                final_decision='Registers'):
    # Decide TM/CM based on linear access
    if num_linear_access >= num_load + num_store:
        data = data[data['Mapping'] == 'CM']
        desired_mapping_from_linear_access = 'CM'
    else:
        TM_static_analysis = data[data['Mapping'] == 'TM']
        # sort based on ALUs
        if TM_static_analysis['ALUs'].max() < 128:
            data = data[data['Mapping'] == 'CM']
            desired_mapping_from_linear_access = 'CM'
        else:
            data = data[data['Mapping'] == 'TM']
            desired_mapping_from_linear_access = 'TM'

    data.to_csv('tmp/'+app_name+'_'+analysis_group_index+'_'+analysis_index+'_linear_access.csv', index=False)

    Static_and_linear_access_configuration = sort_and_filter_data(data, sort_order, filter_number, final_decision)

    return data, Static_and_linear_access_configuration, desired_mapping_from_linear_access


def find_optimum_configuration(app_name, configuration, file_path='input_conf_example.csv'):
    input_data = pd.read_csv(file_path)


    ###########################Add App-specific info############################
    workgroup_size = int(input_data[input_data['app_name'] == app_name]['workgroup size'])
    num_workgroup = int(input_data[input_data['app_name'] == app_name]['num workgroup'])
    
    global_size = workgroup_size * num_workgroup

    # Read the flop/byte ratio
    flop_byte = input_data[input_data['app_name'] == app_name]['Flop/Byte']
    flop_ratio = float(flop_byte.values[0])*4/64 # 4 bytes per float, 64 bytes per cache line
    mem_ratio = (1-flop_ratio)

    # Read the number of divergent branches
    divergent_branches = input_data[input_data['app_name'] == app_name]['Divergent branch TM'].values[0]

    # Decide TM/CM based on linear access
    num_load = input_data[input_data['app_name'] == app_name]['Load(TM)'].values[0]
    num_store = input_data[input_data['app_name'] == app_name]['Store(TM)'].values[0]
    num_linear_access = input_data[input_data['app_name'] == app_name]['TM_linear_access'].values[0]
    ############################################################################
    
    # Read the configuration data
    configuration_data = pd.read_csv('configurations_final.csv')

    # Add Occupancy and Depth columns to the DataFrame (Limit ALUs to 256)
    configuration_data = calculate_occupancy_and_depth(configuration_data, workgroup_size, num_workgroup, flop_ratio, mem_ratio)

    # Apply different 
    after_occupancy_data, _ = apply_occupancy_greedy('0', '0', configuration_data, app_name)
    after_throughput_data, Throughput_greedy_configuration, desired_warp_depth_from_throughput_greedy = apply_throughput_greedy('0', '1', after_occupancy_data, app_name)
    after_divergent_data, Static_and_linear_branch_configuration, desired_warp_width_from_linear_access = apply_divergent_branch_analysis('0', '2', configuration_data, app_name, divergent_branches)
    after_linear_data, Static_and_linear_access_configuration, desired_mapping_from_linear_access = apply_linear_access_analysis('0', '3', configuration_data, app_name, num_load, num_store, num_linear_access)
   
    
   



    ####################################################################################

    # Combining all analysis
    # Apply the first filter and check if the DataFrame becomes empty
    if not after_occupancy_data.empty:
        filtered = after_occupancy_data[after_occupancy_data['Mapping'] == desired_mapping_from_linear_access]
        if not filtered.empty:
            after_occupancy_data = filtered

    # Apply the second filter and check if the DataFrame becomes empty
    if not after_occupancy_data.empty:
        filtered = after_occupancy_data[after_occupancy_data['T'].isin(desired_warp_width_from_linear_access)]
        if not filtered.empty:
            after_occupancy_data = filtered

    # Apply the third filter and check if the DataFrame becomes empty
    if not after_occupancy_data.empty:
        filtered = after_occupancy_data[after_occupancy_data['W'].isin(desired_warp_depth_from_throughput_greedy)]
        if not filtered.empty:
            after_occupancy_data = filtered
    
    static_combined_sort_order = ['C', 'Registers']
    static_combined_filter_number =[2, 1]
    static_combined_final_decision = 'ALUs'

    static_combined_configuration = sort_and_filter_data(after_occupancy_data, static_combined_sort_order, static_combined_filter_number, static_combined_final_decision)
      
    

    """
    # filter by register, L1, L2
    # filter top 4 registers
    unique_registers = pd.Series(area_confined['Registers'].unique())
    area_confined = area_confined[area_confined['Registers'].isin(unique_registers.nlargest(4))]
    # filter top 2 L1
    unique_L1 = pd.Series(area_confined['L1(Dcache)'].unique())
    area_confined = area_confined[area_confined['L1(Dcache)'].isin(unique_L1.nlargest(2))]
    # filter top 1 L2 (2MB)
    unique_L2 = pd.Series(area_confined['L2'].unique())
    area_confined = area_confined[area_confined['L2'].isin(unique_L2.nlargest(1))]

    # filter top 1 total occupancy
    unique_occupancy = pd.Series(area_confined['total_occupancy'].unique())
    area_confined = area_confined[area_confined['total_occupancy'].isin(unique_occupancy.nlargest(1))]

    area_confined = area_confined.sort_values(by='total_occupancy', ascending=False)
    Occupancy_greedy_configuration = (area_confined['C'].values[0], area_confined['W'].values[0], area_confined['T'].values[0], area_confined['L2'].values[0], area_confined['Mapping'].values[0])
    """
    # Temporarily commented out since divergence branch should be more related to warp width, not warp depth
    """
    if divergent_branches.values[0] == 0:
        static_analysis = static_analysis[static_analysis['W'] == 2]
        if global_size > 1024*1024:
            static_analysis = static_analysis[static_analysis['L2'] == 2]
        else:
            static_analysis = static_analysis[static_analysis['L2'] == 1]
    elif divergent_branches.values[0] >= 1 and divergent_branches.values[0] < 5:
        static_analysis = static_analysis[static_analysis['W'] == 4]
        if global_size > 1024:
            static_analysis = static_analysis[static_analysis['L2'] == 2]
        else:
            static_analysis = static_analysis[static_analysis['L2'] == 1]
    else:
        static_analysis = static_analysis[static_analysis['W'] == 8]
        static_analysis = static_analysis[static_analysis['L2'] == 2]
    """

    #static_analysis = static_analysis.sort_values(by='ALUs', ascending=False)
    #Static_analysis_configuration = (static_analysis['C'].values[0], static_analysis['W'].values[0], static_analysis['T'].values[0], static_analysis['L2'].values[0], static_analysis['Mapping'].values[0])

    #static_analysis.to_csv('tmp/'+app_name+'_static_analysis.csv', index=False)


    return Occupancy_greedy_configuration, Throughput_greedy_configuration, Static_and_linear_access_configuration, Static_and_linear_branch_configuration, static_combined_configuration
    
    

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

    try:
        configuration_row = data.loc[(app_name,) + config_tuple]
    except KeyError:
        print("Configuration not found for app:", app_name, "Configuration:", config_tuple)
        return "Configuration not found"
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

    Area_greedy_configuration_cycles = find_cycles_at_given_config(data, app, Area_greedy_configuration)
    Most_common_configuration_cycles = find_cycles_at_given_config(data, app, Most_common_configuration)

    Occupancy_greedy_configuration, Throughput_greedy_configuration, Static_linear_configuration, Static_branch_configuration, Static_combined_configuration = find_optimum_configuration(app, Static_analysis_configuration) 

    Occupancy_greedy_configuration_cycles = find_cycles_at_given_config(data, app, Occupancy_greedy_configuration)
    Throughput_greedy_configuration_cycles = find_cycles_at_given_config(data, app, Throughput_greedy_configuration)

    Static_linear_configuration_cycles = find_cycles_at_given_config(data, app, Static_linear_configuration)
    Static_branch_configuration_cycles = find_cycles_at_given_config(data, app, Static_branch_configuration)
    Static_combined_configuration_cycles = find_cycles_at_given_config(data, app, Static_combined_configuration)



    try:
        Area_greedy_configuration_cycles_relative = int(Area_greedy_configuration_cycles)/int(Best_cycles)
    except:
        Area_greedy_configuration_cycles_relative = "N/A"
    try:
        Most_common_configuration_cycles_relative = int(Most_common_configuration_cycles)/int(Best_cycles)
    except:
        Most_common_configuration_cycles_relative = "N/A"
    try:
        Occupancy_greedy_configuration_cycles_relative = int(Occupancy_greedy_configuration_cycles)/int(Best_cycles)
    except:
        Occupancy_greedy_configuration_cycles_relative = "N/A"
    try:
        Throughput_greedy_configuration_cycles_relative = int(Throughput_greedy_configuration_cycles)/int(Best_cycles)
    except:
        Throughput_greedy_configuration_cycles_relative = "N/A"
    try:
        Static_linear_configuration_cycles_relative = int(Static_linear_configuration_cycles)/int(Best_cycles)
    except:
        Static_linear_configuration_cycles_relative = "N/A"
    try:
        Static_branch_configuration_cycles_relative = int(Static_branch_configuration_cycles)/int(Best_cycles)
    except:
        Static_branch_configuration_cycles_relative = "N/A"
    try:
        Static_combined_configuration_cycles_relative = int(Static_combined_configuration_cycles)/int(Best_cycles)
    except:
        Static_combined_configuration_cycles_relative = "N/A"



    data_to_export.append({
        "App": app,
        "Best configuration": Best_congiruation,
        "Best cycles": Best_cycles,
        "Area greedy configuration": Area_greedy_configuration,
        "Area greedy cycles": Area_greedy_configuration_cycles,
        "Area greedy cycles(relative)": Area_greedy_configuration_cycles_relative,
        "Most common configuration": Most_common_configuration,
        "Most common configuration cycles": Most_common_configuration_cycles,
        "Most common configuration cycles(relative)": Most_common_configuration_cycles_relative,
        "Occupancy greedy configuration": Occupancy_greedy_configuration,
        "Occupancy greedy configuration cycles": Occupancy_greedy_configuration_cycles,
        "Occupancy greedy configuration cycles(relative)": Occupancy_greedy_configuration_cycles_relative,
        "Throughput greedy configuration": Throughput_greedy_configuration,
        "Throughput greedy configuration cycles": Throughput_greedy_configuration_cycles,
        "Throughput greedy configuration cycles(relative)": Throughput_greedy_configuration_cycles_relative,
        "Static+Linear configuration": Static_linear_configuration,
        "Static+Linear configuration cycles": Static_linear_configuration_cycles,
        "Static+Linear configuration cycles(relative)": Static_linear_configuration_cycles_relative,
        "Static+Linear+Branch configuration": Static_branch_configuration,
        "Static+Linear+Branch configuration cycles": Static_branch_configuration_cycles,
        "Static+Linear+Branch configuration cycles(relative)": Static_branch_configuration_cycles_relative,
        "Static+Combined configuration": Static_combined_configuration,
        "Static+Combined configuration cycles": Static_combined_configuration_cycles,
        "Static+Combined configuration cycles(relative)": Static_combined_configuration_cycles_relative
    })



# Now, you can perform lookups using .loc with a tuple representing the values of your keys
# For example, to look up data for a specific combination of keys:
df_to_export = pd.DataFrame(data_to_export)
df_to_export.to_csv(output_file_path, index=False)

# If you need to reset the index to its default integer-based index later, you can do so with:
data.reset_index(inplace=True)
