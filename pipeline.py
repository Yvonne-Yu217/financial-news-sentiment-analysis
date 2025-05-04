#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial News Sentiment Analysis Pipeline

This script orchestrates the complete pipeline of the financial news sentiment analysis system:
1. Crawling Sina Finance news categories
2. Scraping news content
3. Downloading news images
4. Basic image filtering
5. Image clarity analysis
6. Text analysis
7. Image quality processing
8. Sentiment analysis (using pre-trained ViT model)
9. Calculating daily PhotoPes (image sentiment) index
10. Calculating daily TextPes (text sentiment) index  
11. Recalculating quality scores
12. Merging data and calculating returns

Usage:
    python pipeline.py [options]

Options:
    --start-year YEAR    Start year (default: 2014)
    --end-year YEAR      End year (default: 2024)
    --steps STEPS        Comma-separated list of steps to run (default: all)
                         Valid steps: [crawl, scrape, images, filter, clarity, text, quality, 
                                       sentiment, photopes, textpes, recompute, merge]
    --skip-steps STEPS   Comma-separated list of steps to skip
    --mongo-uri URI      MongoDB connection URI (default: mongodb://localhost:27017/)
    --db-name NAME       MongoDB database name (default: sina_news_dataset_test)
    --excel-file FILE    Excel file with market index data (default: Stock Market Index.xlsx)
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)

# Define all steps and their corresponding scripts
PIPELINE_STEPS = {
    'crawl': '1sina_news_category_crawler.py',
    'scrape': '2news_scraper.py',
    'images': '3image_downlowder.py',
    'filter': '4image_basic_filter.py',
    'clarity': '5clarity_analysis_helper.py',
    'text': '6text_analysis_helper.py',
    'quality': '7image_quality_processor.py',
    'sentiment': '9sentiment_analyzer.py',
    'photopes': '10calculate_daily_photopes.py',
    'textpes': '11calculate_daily_textpes.py',
    'recompute': '12recalculate_quality_scores.py',
    'merge': '13merge_data_and_calculate_returns.py'
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Financial News Sentiment Analysis Pipeline')
    parser.add_argument('--start-year', type=int, default=2014, help='Start year for analysis')
    parser.add_argument('--end-year', type=int, default=2024, help='End year for analysis')
    parser.add_argument('--steps', type=str, default='all', 
                        help='Comma-separated list of steps to run')
    parser.add_argument('--skip-steps', type=str, default='', 
                        help='Comma-separated list of steps to skip')
    parser.add_argument('--mongo-uri', type=str, default='mongodb://localhost:27017/',
                        help='MongoDB connection URI')
    parser.add_argument('--db-name', type=str, default='sina_news_dataset_test',
                        help='MongoDB database name')
    parser.add_argument('--excel-file', type=str, default='Stock Market Index.xlsx',
                        help='Excel file with market index data')
    return parser.parse_args()

def run_step(step_name, script_path, args):
    """Run a single pipeline step."""
    start_time = time.time()
    logging.info(f"{'='*20} Starting step: {step_name} {'='*20}")
    
    # Build command with appropriate arguments
    cmd = [sys.executable, script_path]
    
    # Add appropriate arguments based on the step
    if step_name in ['crawl', 'scrape', 'images', 'filter', 'clarity', 'text', 'quality', 'sentiment']:
        cmd.extend(['--start-year', str(args.start_year), '--end-year', str(args.end_year)])
    
    if step_name in ['crawl', 'scrape', 'images']:
        cmd.extend(['--mongo-uri', args.mongo_uri, '--db-name', args.db_name])
    
    if step_name == 'sentiment':
        cmd.extend(['--model', 'improved_vit_sentiment_model.pth'])
    
    if step_name in ['photopes', 'textpes']:
        years = list(range(args.start_year, args.end_year + 1))
        cmd.extend(['--years'] + [str(y) for y in years])
        cmd.extend(['--plot', '--compare'])
        if args.excel_file:
            cmd.extend(['--update-excel', args.excel_file])
    
    if step_name == 'merge':
        # This step requires the Excel file path
        cmd.extend([args.excel_file])
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Output: {result.stdout}")
        if result.stderr:
            logging.warning(f"Errors: {result.stderr}")
        logging.info(f"Step {step_name} completed in {time.time() - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Step {step_name} failed with error code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return False

def ensure_directories():
    """Ensure all required directories exist."""
    required_dirs = ['images', 'results', 'plots']
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

def main():
    """Main pipeline function."""
    args = parse_arguments()
    
    # Determine which steps to run
    steps_to_run = []
    if args.steps.lower() == 'all':
        steps_to_run = list(PIPELINE_STEPS.keys())
    else:
        steps_to_run = [s.strip() for s in args.steps.split(',')]
    
    # Remove skipped steps
    if args.skip_steps:
        skip_steps = [s.strip() for s in args.skip_steps.split(',')]
        steps_to_run = [s for s in steps_to_run if s not in skip_steps]
    
    # Validate steps
    invalid_steps = [s for s in steps_to_run if s not in PIPELINE_STEPS]
    if invalid_steps:
        logging.error(f"Invalid steps: {', '.join(invalid_steps)}")
        logging.error(f"Valid steps are: {', '.join(PIPELINE_STEPS.keys())}")
        return 1
    
    # Ensure required directories exist
    ensure_directories()
    
    # Run the pipeline
    pipeline_start_time = time.time()
    logging.info(f"Starting pipeline with steps: {', '.join(steps_to_run)}")
    logging.info(f"Year range: {args.start_year}-{args.end_year}")
    
    # Check if model file exists for the sentiment step
    if 'sentiment' in steps_to_run and not os.path.exists('improved_vit_sentiment_model.pth'):
        logging.warning("Model file 'improved_vit_sentiment_model.pth' not found.")
        logging.warning("The sentiment analysis step will attempt to download or train the model.")
    
    # Track successful and failed steps
    successful_steps = []
    failed_steps = []
    
    # Execute each step
    for step in steps_to_run:
        script_path = PIPELINE_STEPS[step]
        if not os.path.exists(script_path):
            logging.error(f"Script file not found: {script_path}")
            failed_steps.append(step)
            continue
        
        if run_step(step, script_path, args):
            successful_steps.append(step)
        else:
            failed_steps.append(step)
    
    # Report results
    total_time = time.time() - pipeline_start_time
    logging.info(f"{'='*20} Pipeline Complete {'='*20}")
    logging.info(f"Total execution time: {total_time:.2f} seconds")
    logging.info(f"Successful steps: {len(successful_steps)}/{len(steps_to_run)}")
    
    if successful_steps:
        logging.info(f"Successful: {', '.join(successful_steps)}")
    if failed_steps:
        logging.error(f"Failed: {', '.join(failed_steps)}")
    
    return 0 if not failed_steps else 1

if __name__ == "__main__":
    sys.exit(main())
