from datetime import datetime 
import re
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Function to clean mileage (convert to integer)
def clean_mileage(mileage_str: str):
    if mileage_str.upper().startswith("-"):
        return 0
    # Remove non-numeric characters (like ' and km)
    mileage_clean = re.sub(r"[^\d]", "", mileage_str)
    return int(mileage_clean)

# Function to clean price (convert to float)
def clean_price(price_str):
    price_clean = re.sub(r"[^\d]", "", price_str)
    return float(price_clean)

# Function to clean registration (convert to date)
def clean_registration(registration_str):
    # Convert to a datetime object
    if registration_str.upper().startswith("VEICOLO"):
        return datetime.today().date()
    return datetime.strptime(registration_str, "%m.%Y").date()

def wait_for_page_load(self) -> bool:
    # Wait until the page's readyState is 'complete'
    try:
        WebDriverWait(self.browser, 5).until(
            lambda browser: browser.execute_script("return document.readyState") == "complete"
        )
        print("Page fully loaded.")
        return True
    except:
        print("Page took too long to load.")
        return False

def wait_for_elements(self, xpath, timeout=5):
    """
    Wait for specific elements to load on the page.
"""
    try:
        WebDriverWait(self.browser, timeout).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        print(f"Elements located using XPath: {xpath}")
        return True
    except:
        print(f"Elements not found using XPath: {xpath}")
        return False

class AutoScout24Scraper:
    def __init__(self, make, model, version, year_from, year_to, power_from, power_to, powertype, zip_list, zipr):
        self.make = make
        self.model = model
        self.version = version
        self.year_from = year_from
        self.year_to = year_to
        self.power_from = power_from
        self.power_to = power_to
        self.powertype = powertype
        self.zip_list = zip_list
        self.zipr = zipr
        self.base_url = ("https://www.autoscout24.ch/it/s/mo-{}/mk-{}?priceTo=175000&bodyTypes[0]=suv&bodyTypes[1]=saloon&bodyTypes[2]=coupe&doorsFrom=5&doorsTo=5&seatsFrom=5&seatsTo=5&transmissionTypeGroups[0]=automatic&firstRegistrationYearFrom=2004&firstRegistrationYearTo=2024&sort[0][type]=FIRST_REGISTRATION_DATE&sort[0][order]=ASC")
        self.listing_frame = pd.DataFrame(
            columns=["make", "model", "mileage", "fuel-type", "first-registration", "price"])
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--incognito")
        self.options.add_argument("--ignore-certificate-errors")
        self.browser = webdriver.Chrome(options=self.options)

    def generate_urls(self, num_pages, zip):
        url_list = [self.base_url.format(self.model, self.make)]
        for i in range(1, num_pages, 2 if num_pages > 40 else 1):
            url_to_add = (self.base_url.format(self.model, self.make) + f"&pagination[page]={i}")
            url_list.append(url_to_add)
        return url_list

    def scrape(self, num_pages, verbose=False):
        url_list = []
        #for zip in self.zip_list:
        url_list.extend(self.generate_urls(num_pages, {}))
        

        for webpage in url_list:
            # Usage Example:
            self.browser.get(webpage)
            if not wait_for_page_load(self):
                continue

            total_height = self.browser.execute_script("return document.body.scrollHeight") - 4500
            scroll_position = 0  # Starting scroll position
            scroll_step = 587  # Adjust this for how much you want to scroll at a time (in pixels)

            # Scroll incrementally until we reach the bottom of the page
            while scroll_position < total_height:
                # Scroll by the defined step size
                scroll_position += scroll_step
                self.browser.execute_script(f"window.scrollTo(0, {scroll_position});")

                # Wait for the main listings container to load
                wait_for_elements(self,"//div[contains(@class,'css-e0jgn')]")
                # Once the mileage element is found, we retrieve all the parent listings (with class 'css-2i9fo6')
                listings = self.browser.find_elements(By.XPATH, "//div[contains(@class,'css-2i9fo6')]")

                print("Found " + str(len(listings)) + " listings on page " + webpage + "out of " + str(len(url_list)) + " pages")

                for listing in listings:
                    try :
                        # Mileage (157'500 km)
                        data_mileage = listing.find_element("xpath", ".//div[@class='chakra-stack css-e0jgn']/p").text
                        # Fuel type (Benzina)
                        data_fuel_type = listing.find_element("xpath", ".//div[@class='chakra-stack css-6hoy3o']/p").text
                        # First registration year (10.2011)
                        data_first_registration = listing.find_element("xpath", ".//div[@class='chakra-stack css-15wfwt4']/p").text
                        # Price (CHF 8'500.–)
                        data_price = listing.find_element("xpath", ".//p[@class='chakra-text css-bwl0or']").text


                        listing_data = {
                            "make": self.model,
                            "model": self.make,
                            "mileage": clean_mileage(data_mileage),
                            "fuel-type": data_fuel_type,
                            "first-registration": clean_registration(data_first_registration),
                            "price": clean_price(data_price)
                        }

                        if verbose:
                            print(listing_data)

                        frame = pd.DataFrame(listing_data, index=[0])
                        self.listing_frame = self.listing_frame._append(frame, ignore_index=True)
                    except Exception as e:
                        print("Error for listing n°" + str(listings.index(listing)))

                # Wait a bit to let new content load (if any)
                time.sleep(1)  # Adjust this based on content loading time

                # Check if the total height has changed due to lazy loading
                new_total_height = self.browser.execute_script("return document.body.scrollHeight")
                
                # Update the total height if new content is loaded
                if new_total_height > total_height:
                    total_height = new_total_height

                # Break if we've reached or exceeded the final height of the document
                if scroll_position >= total_height:
                    print("Reached the end of the page.")
                    break
            
            

    def save_to_csv(self, filename="listings.csv"):
        self.listing_frame.to_csv(filename, index=False)
        print("Data saved to", filename)

    def quit_browser(self):
        self.browser.quit()
