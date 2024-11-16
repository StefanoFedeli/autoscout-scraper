import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt


class MileagePricePredictionRegression:
    def __init__(self, mileage_values, years_on_road, average_price_values, std_deviation_values):
        self.mileage_values = mileage_values
        self.years_on_road = years_on_road
        self.average_price_values = average_price_values
        self.std_deviation_values = std_deviation_values

    def do_regression(self, plot=True):
        degrees, rss_scores = self._evaluate_degrees(degrees=range(1, 5))
        if plot:
            self._plot_rss(degrees, rss_scores)
        best_degree = self._select_best_degree(degrees, rss_scores)
        poly_features, poly_reg = self._train_regression(self.mileage_values, self.years_on_road, self.average_price_values, best_degree)
        predicted_prices = self._predict(self.mileage_values, self.years_on_road, poly_features, poly_reg)
        return predicted_prices, best_degree

    def _evaluate_degrees(self, degrees=range(1, 2)):
        k = 10
        rss_scores = []
        for degree in degrees:
            X = np.column_stack((self.mileage_values, self.years_on_road))  # Use both mileage and years
            y = self.average_price_values

            kf = KFold(n_splits=k)
            rss = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                poly_features, poly_reg = self._train_regression(X_train[:, 0], X_train[:, 1], y_train, degree)
                y_pred = self._predict(X_test[:, 0], X_test[:, 1], poly_features, poly_reg)
                rss.append(self._calculate_rss(y_test, y_pred))

            rss_scores.append(np.mean(rss))
        return degrees, rss_scores

    def _predict(self, mileage_values, years_on_road, poly_features, poly_reg):
        X_test = np.column_stack((mileage_values, years_on_road))
        y_pred = poly_reg.predict(poly_features.fit_transform(X_test))
        return y_pred

    def _train_regression(self, mileage_values, years_on_road, price_values, degree):
        X = np.column_stack((mileage_values, years_on_road))
        poly_features = PolynomialFeatures(degree=degree)
        poly_reg = LinearRegression()
        poly_reg.fit(poly_features.fit_transform(X), price_values)
        return poly_features, poly_reg

    def _select_best_degree(self, degrees, rss_scores, verbose=False):
        best_degree = degrees[np.argmin(rss_scores)]
        if verbose:
            print(f"The degree that minimizes RSS is {best_degree}")
        return best_degree

    def _calculate_rss(self, y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2)

    def _plot_rss(self, degrees, rss_scores):
        plt.figure(figsize=(10, 6))
        plt.plot(degrees, rss_scores, marker='o', linestyle='-')
        plt.title('RSS for Different Polynomial Degrees')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('RSS (Mean)')
        plt.grid(True)
        plt.show()

    def calculate_depreciation(self, initial_price, mileage_values, years_on_road, predicted_prices):
        """
        Calculate depreciation percentage from initial price.
        """
        depreciation_percentage = ((initial_price - predicted_prices) / initial_price) * 100
        return depreciation_percentage

    def plot_depreciation(self,depreciation_ideal):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0, 21), depreciation_ideal / 100, label='Depreciation')  # Plot from 0 to 1
        plt.axhline(y=1, color='r', linestyle='--', label='Initial Price (1)')
        plt.xlabel('Years on the Road')
        plt.ylabel('Depreciation Percentage (0 to 1)')
        plt.title('Depreciation vs Years on the Road (Ideal Mileage: 18,000 km/year)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_price_over_years(self,df, predicted_prices_ideal, years_on_road_days):
        """
        Plots how the predicted price moves over the years, along with actual data points from the dataframe.
        """
        # Extract actual mileage and price from the dataframe for plotting
        actual_years_on_road = df['year-on-the-road'].values / 365  # Convert days back to years
        actual_prices = df['price'].values
        
        # Plot the predicted prices over the years (ideal mileage case)
        plt.figure(figsize=(10, 6))
        plt.plot(years_on_road_days / 365, predicted_prices_ideal, label='Predicted Price', color='blue')
        
        # Plot actual data points from the dataframe
        plt.scatter(actual_years_on_road, actual_prices, color='red', label='Actual Prices (AutoScout Data)', s=100, alpha=0.7, edgecolors='black')

        # Add labels and title
        plt.xlabel('Years on the Road')
        plt.ylabel('Price (CHF)')
        plt.title('Predicted Price vs Actual Price Over the Years')
        
        # Show legend
        plt.legend()
        
        # Grid for easier visualization
        plt.grid(True)
        
        # Show the plot
        plt.show()

    def save_prediction_to_csv(self, model_name, make_name, predicted_prices_ideal, depreciation_ideal, years_on_road_days, filename):
        """
        Save the model's depreciation and price prediction information to a CSV, including the new car with 0 years and 0 mileage.
        If the predicted prices or depreciation start to increase, they will be kept constant from the last downward point.
        """
        # Calculate ideal mileage (18,000 km per year, so 9,000 km every 6 months)
        km_per_year = 18000
        ideal_mileage = np.arange(0, len(years_on_road_days)) * (km_per_year / 2)  # Every 6 months, 9,000 km increments
        # Convert years on the road from days to years
        years_on_road = years_on_road_days / 365

        # Ensure prices and depreciation do not increase after starting to decrease
        for i in range(2, len(predicted_prices_ideal)):
            # Check if the predicted price is increasing
            if predicted_prices_ideal[i] > predicted_prices_ideal[i - 1]:
                # Calculate the last downward trend
                last_step_down = predicted_prices_ideal[i - 5] - predicted_prices_ideal[i - 4]
                last_depreciation_step = depreciation_ideal[i - 4] - depreciation_ideal[i - 5]
                
                # Apply the downward trend linearly for the remaining points
                for j in range(i, len(predicted_prices_ideal)):
                    predicted_prices_ideal[j] = predicted_prices_ideal[j - 1] - last_step_down
                    depreciation_ideal[j] = depreciation_ideal[j - 1] + last_depreciation_step

                break  # Stop checking after the trend is corrected
        
        predicted_prices_ideal = [max(0, price) for price in predicted_prices_ideal]
        depreciation_ideal = [min(1, max(0, dep / 100)) for dep in depreciation_ideal]
        # Prepare the data for CSV storage
        data = {
            "Make": [make_name] * len(years_on_road),
            "Model": [model_name] * len(years_on_road),
            "Year_on_Road": years_on_road,
            "Mileage": ideal_mileage,
            "Predicted_Price": predicted_prices_ideal,
            "Depreciation_Percentage": depreciation_ideal   # Storing as a fraction (0 to 1)
        }

        # Save to CSV file
        pd.DataFrame(data).to_csv(filename, index=False)
        print(f"Saved prediction data for {make_name} {model_name} to {filename}")


