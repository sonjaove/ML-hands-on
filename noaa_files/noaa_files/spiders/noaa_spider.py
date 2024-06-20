import os
import scrapy
from urllib.parse import urljoin, urlparse

class NoaaSpider(scrapy.Spider):
    name = "noaa_spider"
    start_urls = [f"https://www.ncei.noaa.gov/data/precipitation-persiann/access/{year}/" for year in range(1983, 2024)]

    def parse(self, response):
        for link in response.css('a::attr(href)').getall():
            file_url = response.urljoin(link)

            # Skip parent directory links and URLs with query parameters
            if file_url == "../" or '?' in file_url:
                continue

            # Ensure the URL points to a file and not a directory
            if file_url.endswith('/'):
                continue

            # Parse the URL to get the file name
            parsed_url = urlparse(file_url)
            file_name = os.path.basename(parsed_url.path)
            file_path = os.path.join("C:\\Users\\Ankit\\Documents\\Vedanshi\\nc_files\\", response.url.split('/')[-2], file_name)

            # Skip invalid filenames
            if not file_name:
                continue
            

            # Yield a new request for the file URL
            if not os.path.exists(file_path):
                yield scrapy.Request(file_url, self.save_file, meta={'file_path': file_path})
            else:
                 self.logger.info(f"Skipping {file_path} as it already exists")

    def save_file(self, response):
        file_name = response.meta['file_name']
        year = response.meta['year']

        # Define the local file path
        location = os.path.join("C:\\Users\\Ankit\\Documents\\Vedanshi\\ML-hands-on\\course_work\\nc_files\\", year)
        file_path = os.path.join(location, file_name)

        # Create the directory if it doesn't exist
        os.makedirs(location, exist_ok=True)

        # Save the file content to the specified directory
        with open(file_path, 'wb') as file:
            file.write(response.body)

        print(f"Downloaded and saved: {file_path}")