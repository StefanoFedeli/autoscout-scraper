import numpy as np
import glob
from Analysis.Comparision import DepreciationPlotter
from Analysis.PredictorFunction import MileagePricePredictionRegression
from Miner.AutoScout24Scraper import AutoScout24Scraper
from Analysis.DataProcessor import DataProcessor
from Analysis.MileagePriceRegression import MileagePriceRegression
from Miner.TextFileHandler import TextFileHandler
from datetime import datetime

import os

MAKE = "vw"
MODEL = "touareg"
SCRAPE = True
PAGES = 21
SINGLE = True
DO_ANALISYS = False
RANGE_LOW = 2
RANGE_HIGH = 8


def main(scrape=False):
    zip_list = where_to_search()
    if scrape:
        scrape_autoscout(zip_list)
    # Data Processing
    data_preprocessed = preprocess(downloaded_listings_file,output_file_preprocessed)
    # Mileage-Price Regression
    #perform_regression(data_preprocessed)
    perform_prediction(data_preprocessed, model=MODEL, make=MAKE, to_plot=True)


def perform_regression(data_preprocessed):
    grouped_data = data_preprocessed.groupby('mileage_grouped')['price'].agg(['mean', 'std']).reset_index()
    mileage_values = grouped_data['mileage_grouped']
    average_price_values = grouped_data['mean']
    std_deviation_values = grouped_data['std']
    regression = MileagePriceRegression(mileage_values, average_price_values, std_deviation_values)
    predicted_prices, best_degree = regression.do_regression()
    # Mileage-Price Plotting
    regression.plot_mileage_price(predicted_prices, best_degree)

def perform_prediction(df,make,model,range_low=0, range_high=20, to_plot=False):
    df = df[(df['year-on-the-road'] >= range_low*365) & (df['year-on-the-road'] <= range_high*365)]
    if df.empty or len(df) <= 10 or df['year-on-the-road'].max() <= ((range_high-range_low)/2) * 365 or df['price'].min() > 25000:
        print(f"No data available for {make} {model}")
        return
    df = df.sort_values(by='year-on-the-road')
    # Extract necessary columns from DataFrame
    mileage_values = df['mileage_grouped'].values
    years_on_road = df['year-on-the-road'].values
    average_price_values = df['price'].values
    # For demonstration, assume the standard deviation is small or constant
    std_deviation_values = np.ones_like(average_price_values) * 500  # Using 500 as a constant deviation for illustration
    # Initialize the regression class with the data
    regressor = MileagePricePredictionRegression(mileage_values, years_on_road, average_price_values, std_deviation_values)
    # Perform regression to find the best degree and get predicted prices
    predicted_prices, best_degree = regressor.do_regression(plot=to_plot)
    print(f"Best degree: {best_degree}") 

    # Ideal mileage: 18,000 km per year
    km_per_year = 18000
    # Generate years in **6-month intervals** (0 to 20 years, every 6 months, in days)
    years_on_road_days = np.arange(RANGE_LOW-1, (RANGE_HIGH+1) * 2) * (365 / 2)  # From 0 to 20 years in 6-month intervals
    # Calculate the ideal mileage for each 6-month interval (0 mileage at 0 years, then increments of 9,000 km every 6 months)
    ideal_mileage = np.arange(RANGE_LOW-1, (RANGE_HIGH+1) * 2) * (km_per_year / 2)
    # Perform prediction on the ideal mileage and years on the road (in days), excluding the first point (0, 0)
    predicted_prices_ideal = regressor._predict(
        ideal_mileage[1:], 
        years_on_road_days[1:], 
        regressor._train_regression(mileage_values, years_on_road, average_price_values, best_degree)[0], 
        regressor._train_regression(mileage_values, years_on_road, average_price_values, best_degree)[1]
    )

    # Assume the initial price for calculating depreciation (e.g., CHF 30,000 for a new car)
    print(max(average_price_values)+2000,predicted_prices_ideal[1],(max(average_price_values)+2000 - predicted_prices_ideal[1])/ (max(average_price_values)+2000) )
    initial_price_delta_1 = (max(average_price_values)+2000 - predicted_prices_ideal[1]) / (max(average_price_values)+2000)
    initial_price = np.quantile(average_price_values, 0.99) if initial_price_delta_1 > 0.2 else max(predicted_prices_ideal)+2000
    # Insert the initial price at the beginning for 0 mileage and 0 years
    predicted_prices_ideal = np.insert(predicted_prices_ideal, 0, initial_price)
    # Calculate depreciation percentage, starting from 0%
    depreciation_ideal = regressor.calculate_depreciation(initial_price, ideal_mileage, years_on_road_days, predicted_prices_ideal)
    # Plot price movement over the years (6-month intervals) with actual data points
    if to_plot:
        regressor.plot_price_over_years(df, predicted_prices_ideal, years_on_road_days)

    # Save the data to a CSV file for later analysis
    regressor.save_prediction_to_csv(model, make, predicted_prices_ideal, depreciation_ideal, years_on_road_days, f'predictions/depreciation_{make}_{model}.csv')

def get_days_on_the_road(registration_date):
    try:
       return (datetime.today().date() - datetime.strptime(registration_date,"%Y-%m-%d").date()).days
    except:
        return 0

def preprocess(input_csv: str, output_csv:str):
    processor = DataProcessor(input_csv)
    data = processor.read_data()
    data['year-on-the-road'] = data['first-registration'].map(get_days_on_the_road)
    data_no_duplicates = processor.remove_duplicates(data)
    data_preprocessed = processor.preprocess_data(data_no_duplicates)
    data_rounded = processor.round(data_preprocessed, 1000)
    processor.save_processed_data(data_rounded, output_csv)
    return data_preprocessed


def scrape_autoscout(zip_list):
    scraper = AutoScout24Scraper(make, model, version, year_from, year_to, power_from, power_to, powertype, zip_list,
                                 zipr)
    scraper.scrape(num_pages, False)
    scraper.save_to_csv(downloaded_listings_file)
    scraper.quit_browser()


def where_to_search():
    handler = TextFileHandler(zip_list_file_path)
    handler.load_data_csv()
    zip_list = handler.export_capoluogo_column()
    zip_list = [item.lower() for item in zip_list]
    return zip_list

def analysis():
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join("listings", "*.csv"))
    # Preprocess each CSV file
    for file_csv in csv_files:
        # Extract filename without path
        filename = os.path.basename(file_csv)
        # Extract make and model from filename pattern "listings_{make}_{model}.csv"
        if filename.startswith("listings_") and filename.endswith(".csv"):
            # Remove 'listings_' from the beginning and '.csv' from the end
            name_parts = filename[:-4].split('_')
            if len(name_parts) >= 2:
                make = name_parts[1]
                model = name_parts[2]   # Handle models that have multiple words
                print(f"Preprocessing file: {file_csv} (Make: {make}, Model: {model})")
                perform_prediction(preprocess(file_csv,f'listings-final/listings_{make}_{model}_preprocessed.csv'),make,model,RANGE_LOW,RANGE_HIGH)
            else:
                print(f"Invalid file format for: {filename}")


def show_depreciation():
    # Initialize the DepreciationPlotter with the folder containing the CSVs
    plotter = DepreciationPlotter('predictions', RANGE_LOW, RANGE_HIGH)
    plotter.load_data()
    plotter.plot_depreciation()
    plotter.plot_derivatives()



if __name__ == "__main__":
    make = MAKE
    model = MODEL
    version = ""
    year_from = "2004"
    year_to = "2024"
    power_from = ""
    power_to = ""
    powertype = ""
    num_pages = PAGES
    zipr = 100

    zip_list_file_path = 'Miner/capoluoghi.csv'
    downloaded_listings_file = f'listings/listings_{make}_{model}.csv'
    output_file_preprocessed = f'listings-final/listings_{make}_{model}_preprocessed.csv'
    # Create the "listings" folder if it doesn't exist
    if not os.path.exists("listings"):
        os.makedirs("listings")

    if SINGLE:
        main(scrape=SCRAPE)
    else:
        analysis() if DO_ANALISYS else None
        show_depreciation()
