import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

class DepreciationPlotter:
    def __init__(self, csv_folder_path, range_low, range_high):
        """
        Initializes the DepreciationPlotter with a folder path where CSV files are stored.
        """
        self.csv_folder_path = csv_folder_path
        self.df = None  # DataFrame to hold all loaded data
        self.range_low = range_low
        self.range_high = range_high
    
    def load_data(self):
        """
        Load all CSV files in the folder into a single DataFrame.
        """
        all_files = glob.glob(self.csv_folder_path + "/*.csv")
        
        # List to hold individual DataFrames
        df_list = []

        for file in all_files:
            df = pd.read_csv(file)
            df_list.append(df)
        
        # Combine all DataFrames into one
        self.df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {len(all_files)} models from {self.csv_folder_path}")
    
    def get_top_x_models(self, head):
        """
        Get the top X models with the least mean depreciation between the 3rd year and the 12th year.
        """
        # Filter the data to include only the years between the 3rd and 12th year
        filtered_df = self.df[(self.df['Year_on_Road'] >= 2) & (self.df['Year_on_Road'] <= 10)]
        
        # Calculate the mean depreciation for each model between the 3rd and 12th year
        mean_depreciation = filtered_df.groupby(['Make', 'Model'])['Depreciation_Percentage'].mean().reset_index()
        
        # Sort by least mean depreciation and get the top X models
        top_x_models = mean_depreciation.sort_values(by='Depreciation_Percentage', ascending=True).head(head)
        
        return top_x_models
    
    def plot_depreciation(self):
        """
        Plot the depreciation percentage over the years for all models, highlight the top 10 in special colors,
        highlight the area under the baseline in light green, and limit the axis to 0-1 for depreciation and 0-20 for years.
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        # Get the top 10 and top 50 models with the least depreciation
        top_50 = self.get_top_x_models(50)
        top_10 = top_50.head(10)
        top_10_models = list(zip(top_10['Make'], top_10['Model']))
        top_50_models = list(zip(top_50['Make'], top_50['Model']))
        
        plt.figure(figsize=(12, 8))
        
        # Highlight area under the baseline depreciation curve (0 to 1 in 20 years)
        baseline_years = range(self.range_low -1, self.range_high + 1)
        baseline_depreciation = [x / self.range_high for x in baseline_years]
        plt.fill_between(baseline_years, baseline_depreciation, color='lightgreen', alpha=0.3, label='Best Area (Under Baseline)')

        # Plot depreciation curves for each model
        for _, model_data in self.df.groupby(['Make', 'Model']):
            make = model_data['Make'].iloc[0]
            model = model_data['Model'].iloc[0]
            label = f"{make} {model}"
            
            # Check if the model is in the top 10
            if (make, model) in top_10_models:
                plt.plot(model_data['Year_on_Road'], model_data['Depreciation_Percentage'], label=label, linewidth=2.5)
            elif (make, model) in top_50_models:
                plt.plot(model_data['Year_on_Road'], model_data['Depreciation_Percentage'], linestyle='--', alpha=0.7)
        
        # Plot baseline depreciation curve (linear depreciation from 0 to 1 over 20 years)
        plt.plot(baseline_years, baseline_depreciation, label='Baseline Depreciation (0 to 1 in 20 years)', color='black', linestyle='--', linewidth=2)
        
        # Limit the axes
        plt.xlim(self.range_low - 1, self.range_high)  # Limit X-axis from 0 to 20 years
        plt.ylim(0, 1)   # Limit Y-axis from 0 to 1 (0% to 100% depreciation)

        # Plot settings
        plt.title('Depreciation Percentage Over the Years for Different Car Models (Top 50)')
        plt.xlabel('Years on the Road')
        plt.ylabel('Depreciation Percentage')
        plt.grid(True)
        plt.legend(loc='best', fontsize='small')
        
        # Show plot
        plt.show()

    def plot_derivatives(self):
        """
        Plot the first derivative (rate of depreciation) of each model between 2 and 12 years on the road.
        Highlight the top 7 models that depreciate the least with special traits.
        Include a baseline depreciation curve for linear depreciation from 0 to 1 in 20 years, 
        and highlight the area under the baseline in light green.
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        # Filter data for years between 2 and 12 years
        filtered_df = self.df[(self.df['Year_on_Road'] >= self.range_low) & (self.df['Year_on_Road'] <= self.range_high)]
        
        # Create a dictionary to store the first derivatives for each model
        model_derivatives = {}

        # Loop through each model and calculate the first derivative of depreciation over time
        for (make, model), model_data in filtered_df.groupby(['Make', 'Model']):
            years_on_road = model_data['Year_on_Road'].values
            depreciation = model_data['Depreciation_Percentage'].values

            # Compute the first derivative (rate of change) using numpy's gradient
            derivative = np.gradient(depreciation, years_on_road)

            # Skip the model if it has any derivative values of 0
            if np.any(derivative == 0):
                continue
            
            # Store the mean derivative for each model
            model_derivatives[(make, model)] = np.mean(derivative)

        # Sort models by the least depreciation (smallest derivative)
        top_25_models = sorted(model_derivatives.items(), key=lambda x: x[1])[:25]
        top_7_models = top_25_models[:7]

        # Prepare plot
        plt.figure(figsize=(12, 8))

        # Plot top 7 models with special traits
        for (make, model), _ in top_7_models:
            model_data = filtered_df[(filtered_df['Make'] == make) & (filtered_df['Model'] == model)]
            years_on_road = model_data['Year_on_Road'].values
            depreciation = model_data['Depreciation_Percentage'].values
            derivative = np.gradient(depreciation, years_on_road)
            label = f"{make} {model}"
            plt.plot(years_on_road, derivative, label=label, linewidth=2.5)

        # Plot the remaining top 25 models with a different style
        for (make, model), _ in top_25_models[7:]:
            model_data = filtered_df[(filtered_df['Make'] == make) & (filtered_df['Model'] == model)]
            years_on_road = model_data['Year_on_Road'].values
            depreciation = model_data['Depreciation_Percentage'].values
            derivative = np.gradient(depreciation, years_on_road)
            label = f"{make} {model}"
            plt.plot(years_on_road, derivative, linestyle='--', label=label, alpha=0.7)

        # Plot settings
        plt.title('First Derivative of Depreciation Between 2 and 12 Years (Top 7 Highlighted)')
        plt.xlabel('Years on the Road')
        plt.ylabel('Depreciation Rate (First Derivative)')
        plt.grid(True)
        plt.legend(loc='best', fontsize='small')

        # Show plot
        plt.show()