import os
import scrapy
from urllib.parse import urljoin, urlparse
from scrapy import signals
from pydispatch import dispatcher
from tqdm import tqdm

class NoaaSpider(scrapy.Spider):
    name = "noaa_spider"
    start_urls = [f"https://www.ncei.noaa.gov/data/precipitation-persiann/access/{year}/" for year in range(1983, 2024)]
    
    def __init__(self, *args, **kwargs):
        super(NoaaSpider, self).__init__(*args, **kwargs)
        self.file_urls = []
        self.pbar = None
        dispatcher.connect(self.spider_closed, signals.spider_closed)
        dispatcher.connect(self.spider_opened, signals.spider_opened)
    
    def spider_opened(self, spider):
        # Initialize the progress bar when the spider is opened
        self.pbar = tqdm(total=0, desc='Processing Files', unit='file')
    
    def spider_closed(self, spider):
        # Close the progress bar when the spider is closed
        if self.pbar:
            self.pbar.close()
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE".center(60))
        print("="*60 + "\n")

    def parse(self, response):
        file_urls = []
        for link in response.css('a::attr(href)').getall():
            file_url = response.urljoin(link)

            # Skip parent directory links and URLs with query parameters
            if file_url.endswith('../') or '?' in file_url:
                continue

            # Ensure the URL points to a file and not a directory
            if file_url.endswith('/'):
                continue

            # Parse the URL to get the file name
            parsed_url = urlparse(file_url)
            file_name = os.path.basename(parsed_url.path)
            year = response.url.split('/')[-2]
            file_path = os.path.join(r"F:\TANISHQ\PERCDR data", year, file_name)

            # Skip invalid filenames
            if not file_name:
                continue

            # Yield a new request for the file URL
            if not os.path.exists(file_path):
                file_urls.append(file_url)
                yield scrapy.Request(file_url, self.save_file, meta={'file_path': file_path})

        # Update progress bar total files to process
        if self.pbar:
            self.pbar.total += len(file_urls)

    def save_file(self, response):
        file_path = response.meta['file_path']

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the file content to the specified directory
        with open(file_path, 'wb') as file:
            file.write(response.body)

        # Update the progress bar
        if self.pbar:
            self.pbar.update(1)

        self.logger.info(f"Downloaded and saved: {file_path}")
