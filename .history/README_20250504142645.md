# Financial News Sentiment Analysis System

This project implements a comprehensive pipeline for analyzing sentiment in financial news from Sina Finance, with a focus on generating sentiment indices based on both text and image content.

## Overview

The system implements a complete pipeline for financial news sentiment analysis:

1. **Data Collection**: Crawling and scraping news articles from Sina Finance.
2. **Image Processing**: Downloading, filtering, and analyzing images from news articles.
3. **Sentiment Analysis**: Analyzing sentiment in both text content and news images.
4. **Index Generation**: Calculating weighted sentiment indices (TextPes and PhotoPes).
5. **Financial Analysis**: Combining sentiment indices with market data for analysis.

## Features

- Multi-threaded and asynchronous data collection
- Quality-based image filtering pipeline
- Deep learning-based image sentiment analysis (Vision Transformer)
- NLP-based text sentiment analysis (RoBERTa)
- Weighted sentiment indices calculation
- Integration with financial market data

## Requirements

- Python 3.8+
- MongoDB
- PyTorch 
- Transformers
- Various Python packages (see `requirements.txt`)

## Pipeline Structure

The system consists of 13 interconnected scripts that form a complete pipeline:

1. `1sina_news_category_crawler.py` - Crawls Sina Finance news categories 
2. `2news_scraper.py` - Scrapes news content 
3. `3image_downlowder.py` - Downloads news images
4. `4image_basic_filter.py` - Performs basic filtering of images
5. `5clarity_analysis_helper.py` - Analyzes image clarity
6. `6text_analysis_helper.py` - Analyzes text content in images
7. `7image_quality_processor.py` - Processes image quality 
8. `8vit_transferlearning.py` - Trains a Vision Transformer model for sentiment analysis
9. `9sentiment_analyzer.py` - Analyzes sentiment in images
10. `10calculate_daily_photopes.py` - Calculates daily PhotoPes index
11. `11calculate_daily_textpes.py` - Calculates daily TextPes index
12. `12recalculate_quality_scores.py` - Recalculates quality scores
13. `13merge_data_and_calculate_returns.py` - Merges data and calculates returns

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start MongoDB:
   ```
   mongod --dbpath /path/to/data/directory
   ```

3. Run the entire pipeline:
   ```
   python pipeline.py
   ```

4. Or run specific steps:
   ```
   python pipeline.py --steps "crawl,scrape,images,sentiment,photopes,textpes,merge" --start-year 2020 --end-year 2024
   ```

## Configuration

The pipeline can be configured using command-line arguments:

- `--start-year`: Start year for data collection (default: 2014)
- `--end-year`: End year for data collection (default: 2024)
- `--steps`: Comma-separated list of pipeline steps to run
- `--skip-steps`: Comma-separated list of pipeline steps to skip
- `--mongo-uri`: MongoDB connection URI
- `--db-name`: MongoDB database name
- `--excel-file`: Excel file with market index data

## Directory Structure

- `/images`: Downloaded news images
- `/results`: Output data files
- `/plots`: Generated charts and visualizations

## Model Files

- `improved_vit_sentiment_model.pth`: Pre-trained Vision Transformer model for image sentiment analysis

## Output Files

- `weighted_photopes.csv`: Daily PhotoPes sentiment index
- `weighted_textpes.csv`: Daily TextPes sentiment index
- `merged_market_sentiment_data.csv`: Merged sentiment and market data

## Notes

- The system requires significant disk space for storing images
- MongoDB is used for data storage and processing
- GPU acceleration is recommended for the sentiment analysis step

## License

MIT License
