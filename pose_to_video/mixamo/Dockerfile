# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update
# Install the necessary dependencies for pyppeteer
RUN apt-get install -y --no-install-recommends \
        wget \
        unzip \
        libx11-xcb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxtst6 \
        libnss3 \
        libcups2 \
        libxss1 \
        libxrandr2 \
        libasound2 \
        libpangocairo-1.0-0 \
        libatk1.0-0 \
        libgtk-3-0
# Install OpenCV
RUN apt-get update && apt-get install -y libopencv-dev python3-opencv
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*



# Install the required packages
RUN pip install Pillow tqdm pyppeteer opencv-python

# Set environment variables
ENV PYPPETEER_HOME /app/.pyppeteer
RUN python -c "import pyppeteer; pyppeteer.chromium_downloader.download_chromium()"

COPY src src