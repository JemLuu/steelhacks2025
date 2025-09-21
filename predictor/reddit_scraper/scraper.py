#!/usr/bin/env python3
"""
Reddit Top Posts Scraper

This script scrapes the top 100 posts from a subreddit's "top this year" section
and saves the post titles to a JSON file using web scraping (no API credentials required).

Usage:
    python scraper.py <subreddit_name>

Example:
    python scraper.py programming

Requirements:
    - requests library (pip install requests)
    - beautifulsoup4 library (pip install beautifulsoup4)
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup


class RedditScraper:
    """Reddit scraper class for fetching top posts from subreddits using web scraping."""

    def __init__(self):
        """Initialize the web scraper with headers to mimic a browser."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def fetch_top_posts(self, subreddit_name: str, limit: int = 100) -> List[str]:
        """
        Fetch the top posts from a subreddit for the current year using web scraping.

        Args:
            subreddit_name: Name of the subreddit to scrape
            limit: Number of posts to fetch (default: 100)

        Returns:
            List of strings in format "title | text"

        Raises:
            Various exceptions for different error conditions
        """
        try:
            print(f"Fetching top {limit} posts from r/{subreddit_name} (top this year)...")

            posts_content = []
            posts_fetched = 0
            after = None  # For pagination

            # Calculate how many pages we need (Reddit shows ~25 posts per page)
            posts_per_page = 25
            pages_needed = (limit + posts_per_page - 1) // posts_per_page

            for page in range(pages_needed):
                # Use Reddit's JSON API endpoint for more reliable data
                url = f"https://www.reddit.com/r/{subreddit_name}/top.json"
                params = {
                    't': 'year',  # top this year
                    'limit': min(100, limit - posts_fetched)  # Reddit's max is 100 per request
                }

                if after:
                    params['after'] = after

                try:
                    # Make request with retry logic
                    response = self._make_request(url, params)

                    # Parse JSON response
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        # If JSON parsing fails, might be an error page
                        if self._check_for_errors_in_text(response.text, subreddit_name):
                            break
                        raise ValueError(f"Unable to parse Reddit response for r/{subreddit_name}")

                    # Check for errors in JSON response
                    if 'error' in data:
                        error_code = data.get('error')
                        if error_code == 403:
                            raise ValueError(f"Access denied to r/{subreddit_name}. The subreddit may be private or banned.")
                        elif error_code == 404:
                            raise ValueError(f"Subreddit 'r/{subreddit_name}' not found. Please check the subreddit name.")
                        else:
                            raise ValueError(f"Reddit API error: {data.get('message', 'Unknown error')}")

                    # Extract posts from JSON data
                    posts_data = data.get('data', {}).get('children', [])

                    if not posts_data:
                        print("No more posts found.")
                        break

                    # Extract titles and text from posts
                    page_posts = []
                    for post_item in posts_data:
                        if posts_fetched >= limit:
                            break

                        post_data = post_item.get('data', {})

                        # Skip stickied posts and ads
                        if post_data.get('stickied') or post_data.get('promoted'):
                            continue

                        title = post_data.get('title', '').strip()
                        selftext = post_data.get('selftext', '').strip()

                        # Clean up the selftext (remove markdown and excessive whitespace)
                        if selftext:
                            selftext = selftext.replace('\n', ' ').replace('\r', ' ')
                            selftext = ' '.join(selftext.split())  # Remove multiple spaces

                        if title:
                            # Format as "title | text" or just "title" if no text
                            if selftext:
                                combined = f"{title} | {selftext}"
                            else:
                                combined = f"{title} | [no text content]"
                            page_posts.append(combined)
                            posts_fetched += 1

                    posts_content.extend(page_posts)

                    # Progress indicator
                    if posts_fetched % 10 == 0 and posts_fetched > 0:
                        print(f"Fetched {posts_fetched} posts...")

                    # Get the "after" parameter for next page from JSON
                    after = data.get('data', {}).get('after')

                    if not after or posts_fetched >= limit:
                        break

                    # Be respectful with requests
                    time.sleep(1)

                except requests.exceptions.RequestException as e:
                    if page == 0:  # If first page fails, it's likely a real error
                        raise
                    else:  # If later pages fail, we can continue with what we have
                        print(f"Warning: Failed to fetch page {page + 1}, continuing with {posts_fetched} posts")
                        break

            print(f"Successfully fetched {posts_fetched} posts from r/{subreddit_name}")
            return posts_content

        except requests.exceptions.RequestException as e:
            if "404" in str(e) or "Not Found" in str(e):
                raise ValueError(f"Subreddit 'r/{subreddit_name}' not found. Please check the subreddit name.")
            elif "403" in str(e) or "Forbidden" in str(e):
                raise ValueError(f"Access denied to r/{subreddit_name}. The subreddit may be private or banned.")
            else:
                raise ConnectionError(f"Network error occurred: {e}")

        except Exception as e:
            raise ConnectionError(f"Unexpected error occurred: {e}")

    def _make_request(self, url: str, params: Dict = None) -> requests.Response:
        """Make a request with error handling and retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

    def _check_for_errors_in_text(self, text: str, subreddit_name: str) -> bool:
        """Check if the response text indicates an error (private, banned, or non-existent subreddit)."""
        # Check for error messages
        error_messages = [
            "this community has been banned",
            "this community is private",
            "there doesn't seem to be anything here",
            "this community doesn't exist",
            "sorry, there doesn't seem to be anything here"
        ]

        page_text = text.lower()
        for error_msg in error_messages:
            if error_msg in page_text:
                if "banned" in error_msg:
                    raise ValueError(f"Subreddit 'r/{subreddit_name}' has been banned.")
                elif "private" in error_msg:
                    raise ValueError(f"Subreddit 'r/{subreddit_name}' is private.")
                else:
                    raise ValueError(f"Subreddit 'r/{subreddit_name}' not found. Please check the subreddit name.")

        # Check if we're redirected to search page
        if "search results" in page_text and "no results" in page_text:
            raise ValueError(f"Subreddit 'r/{subreddit_name}' not found. Please check the subreddit name.")

        return False

    def _check_for_errors(self, soup: BeautifulSoup, subreddit_name: str) -> bool:
        """Check if the page indicates an error (private, banned, or non-existent subreddit)."""
        # Check for error messages
        error_messages = [
            "this community has been banned",
            "this community is private",
            "there doesn't seem to be anything here",
            "this community doesn't exist",
            "sorry, there doesn't seem to be anything here"
        ]

        page_text = soup.get_text().lower()
        for error_msg in error_messages:
            if error_msg in page_text:
                if "banned" in error_msg:
                    raise ValueError(f"Subreddit 'r/{subreddit_name}' has been banned.")
                elif "private" in error_msg:
                    raise ValueError(f"Subreddit 'r/{subreddit_name}' is private.")
                else:
                    raise ValueError(f"Subreddit 'r/{subreddit_name}' not found. Please check the subreddit name.")

        # Check if we're redirected to search page
        if "search results" in page_text and "no results" in page_text:
            raise ValueError(f"Subreddit 'r/{subreddit_name}' not found. Please check the subreddit name.")

        return False

    def _is_sticky_or_ad(self, post_element) -> bool:
        """Check if a post is stickied or an advertisement."""
        # Check for sticky indicators
        if post_element.find('span', {'class': 'stickied-tagline'}):
            return True

        # Check for promoted/ad indicators
        if post_element.find('span', string=lambda x: x and 'promoted' in x.lower()):
            return True

        # Check for data attributes that indicate sticky/promoted
        if post_element.get('data-promoted') or post_element.get('data-stickied'):
            return True

        return False

    def _get_next_page_token(self, soup: BeautifulSoup) -> str:
        """Extract the 'after' token for pagination."""
        next_button = soup.find('span', {'class': 'next-button'})
        if next_button:
            link = next_button.find('a')
            if link and link.get('href'):
                href = link.get('href')
                # Extract the 'after' parameter from the URL
                if 'after=' in href:
                    after_start = href.find('after=') + 6
                    after_end = href.find('&', after_start)
                    if after_end == -1:
                        after_end = len(href)
                    return href[after_start:after_end]
        return None

    def save_to_json(self, subreddit_name: str, posts_content: List[str], filename: str = "to_label.txt") -> None:
        """
        Save the posts with titles and text to a JSON file (appends to existing file).

        Args:
            subreddit_name: Name of the subreddit
            posts_content: List of posts in format "title | text"
            filename: Output filename (default: "to_label.txt")
        """
        try:
            # Check if file exists and load existing data
            data = {}
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"Appending to existing file '{filename}'...")
                except (json.JSONDecodeError, IOError):
                    print(f"Warning: Could not read existing file '{filename}', will create new file.")
                    data = {}

            # Add or update the subreddit data
            if subreddit_name in data:
                print(f"Warning: Subreddit 'r/{subreddit_name}' already exists in file. Overwriting its data...")

            data[subreddit_name] = posts_content

            # Save the combined data
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            total_subreddits = len(data)
            print(f"Successfully saved {len(posts_content)} posts from r/{subreddit_name} to '{filename}'")
            print(f"File now contains data from {total_subreddits} subreddit(s)")

        except IOError as e:
            raise IOError(f"Error writing to file '{filename}': {e}")

        except Exception as e:
            raise Exception(f"Unexpected error while saving file: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scrape top posts from a Reddit subreddit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py programming
  python scraper.py MachineLearning
  python scraper.py python

Note: Make sure to set up your Reddit API credentials before running this script.
See the setup instructions for more details.
        """
    )

    parser.add_argument(
        'subreddit',
        help='Name of the subreddit to scrape (without r/ prefix)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Number of posts to fetch (default: 100, max: 1000)'
    )

    parser.add_argument(
        '--output',
        default='to_label.txt',
        help='Output filename (default: to_label.txt)'
    )

    return parser.parse_args()


def main():
    """Main function to run the Reddit scraper."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Validate arguments
        if args.limit <= 0 or args.limit > 1000:
            print("Error: Limit must be between 1 and 1000")
            sys.exit(1)

        # Clean subreddit name (remove r/ prefix if present)
        subreddit_name = args.subreddit.lower().replace('r/', '')

        if not subreddit_name:
            print("Error: Please provide a valid subreddit name")
            sys.exit(1)

        # Initialize scraper and fetch posts
        scraper = RedditScraper()
        posts_content = scraper.fetch_top_posts(subreddit_name, args.limit)

        if not posts_content:
            print(f"Warning: No posts found in r/{subreddit_name}")
            return

        # Save results to file
        scraper.save_to_json(subreddit_name, posts_content, args.output)

        print(f"\nScraping completed successfully!")
        print(f"Subreddit: r/{subreddit_name}")
        print(f"Posts fetched: {len(posts_content)}")
        print(f"Output file: {args.output}")

    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        sys.exit(1)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    except ConnectionError as e:
        print(f"Connection Error: {e}")
        sys.exit(1)

    except IOError as e:
        print(f"File Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()