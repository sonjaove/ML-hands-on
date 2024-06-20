import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

parent_url = "https://www.ncei.noaa.gov/data/precipitation-persiann/access/"

for year in range(1983, 2024):
    url = parent_url + str(year) + "/"
    response = requests.get(url)
    
    if response.status_code == 200:
        try:
            # Parse the HTML response
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            for link in links:
                file_url = link['href']
                
                # Skip parent directory links and URLs with query parameters
                if file_url == "../" or '?' in file_url:
                    continue
                
                # Construct the full file URL if it's a relative URL
                if not file_url.startswith('http'):
                    file_url = urljoin(url, file_url)
                
                # Ensure the URL points to a file and not a directory
                if file_url.endswith('/'):
                    continue
                
                # Parse the URL to get the file name
                parsed_url = urlparse(file_url)
                file_name = os.path.basename(parsed_url.path)
                
                # Skip invalid filenames
                if not file_name:
                    continue
                
                # Ensure the file URL points to a valid file
                file_response = requests.get(file_url)
                
                if file_response.status_code == 200:
                    # Define the local file path
                    location = os.path.join("C:\\Users\\Ankit\\Documents\\Vedanshi\\ML-hands-on\\course_work\\nc_files\\", str(year))
                    file_path = os.path.join(location, file_name)
                    
                    # Create the directory if it doesn't exist
                    os.makedirs(location, exist_ok=True)
                    
                    # Save the file content to the specified directory
                    with open(file_path, 'wb') as file:
                        file.write(file_response.content)
                    
                    print(f"Downloaded and saved: {file_path}")
                else:
                    print(f"Failed to download {file_url}: {file_response.status_code}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"Failed to retrieve data for year {year}: {response.status_code}")
        print("Response content:", response.text)
