#!/usr/bin/env python3
"""
Twitter scraper using Twikit library
Usage: python twitter_scraper.py <username> [max_tweets]
Example: python twitter_scraper.py jiaseedpudding 50
"""

import sys
import json
import pandas as pd
import asyncio
import os
from urllib.parse import urlparse
from twikit import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_username_from_url(url_or_username):
    """Extract Twitter username from URL or return username if already provided"""
    if url_or_username.startswith('http'):
        parsed = urlparse(url_or_username)
        path = parsed.path.strip('/')
        username = path.split('/')[0]
        return username.replace('@', '')
    else:
        return url_or_username.replace('@', '')

async def login_to_twitter(client, force_login=False):
    """Login to Twitter and save/load cookies"""
    if not force_login:
        try:
            # Try to load existing cookies first
            client.load_cookies(path='cookies.json')
            print("Loaded existing cookies")
            return True
        except:
            print("No existing cookies found.")

    # Try to get credentials from environment variables
    twitter_username = os.getenv('TWITTER_USERNAME')
    twitter_password = os.getenv('TWITTER_PASSWORD')
    twitter_email = os.getenv('TWITTER_EMAIL')

    if not twitter_username or not twitter_password:
        print("No credentials found in .env file.")
        print("Please provide your Twitter credentials:")
        twitter_username = input("Username: ")
        twitter_email = input("Email (if different from username): ") or twitter_username
        twitter_password = input("Password: ")
    else:
        print("Using credentials from .env file")
        if not twitter_email:
            twitter_email = twitter_username

    try:
        # Try with email first (Twitter often requires email for login)
        print("Attempting login...")
        await client.login(
            auth_info_1=twitter_username,
            auth_info_2=twitter_email,
            password=twitter_password
        )
        client.save_cookies('cookies.json')
        print("Login successful! Cookies saved.")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"Login failed: {e}")

        if "366" in error_msg:
            print("\nThis error often indicates:")
            print("1. 2FA is enabled - you may need to provide verification code")
            print("2. Suspicious login detected - try logging in via web browser first")
            print("3. Account may be locked or restricted")
            print("4. Email verification required")
        elif "399" in error_msg:
            print("Phone number verification required")
        elif "64" in error_msg:
            print("Account suspended")

        return False

async def scrape_user_tweets(username, max_tweets=50):
    """Scrape tweets from a specific user"""
    client = Client('en-US')

    # Login to Twitter
    if not await login_to_twitter(client):
        return []

    try:
        # Get user by screen name
        print(f"Fetching user: @{username}")
        user = await client.get_user_by_screen_name(username)

        # Get tweets from the user
        print(f"Scraping up to {max_tweets} tweets...")
        tweets = await user.get_tweets('Tweets', count=max_tweets)

        tweets_data = []
        for tweet in tweets:
            tweets_data.append({
                'created_at': tweet.created_at,
                'favorite_count': tweet.favorite_count,
                'retweet_count': getattr(tweet, 'retweet_count', 0),
                'full_text': tweet.full_text,
                'tweet_id': tweet.id,
                'username': username
            })

        print(f"Successfully scraped {len(tweets_data)} tweets from @{username}")
        return tweets_data

    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg:
            print("403 Forbidden - cookies may have expired. Try re-authenticating...")
            print("Delete cookies.json and run again, or use a fresh Twitter account.")
        elif "429" in error_msg:
            print("Rate limited - wait a few minutes before trying again")
        elif "401" in error_msg:
            print("Unauthorized - authentication failed")
        else:
            print(f"Error scraping tweets from @{username}: {e}")
        return []

def append_to_golden_labels(username, tweets_data):
    """Append scraped tweets to golden_labels.txt in the specified format"""
    try:
        # Extract just the text from tweets
        tweet_texts = [tweet['full_text'] for tweet in tweets_data]

        # Try to read existing file
        try:
            with open('golden_labels.txt', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    existing_data = eval(content)
                else:
                    existing_data = {}
        except (FileNotFoundError, SyntaxError):
            existing_data = {}

        # Add new data
        existing_data[username] = tweet_texts

        # Write back to file in the specified format
        with open('golden_labels.txt', 'w', encoding='utf-8') as f:
            f.write(str(existing_data))

        print(f"Successfully appended {len(tweet_texts)} tweets for @{username} to golden_labels.txt")

    except Exception as e:
        print(f"Error writing to file: {e}")

def save_detailed_data(username, tweets_data):
    """Save detailed tweet data to CSV and JSON for analysis"""
    if not tweets_data:
        return

    # Create DataFrame
    df = pd.DataFrame(tweets_data)

    # Save to CSV
    csv_filename = f'{username}_tweets.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Detailed data saved to {csv_filename}")

    # Save to JSON
    json_filename = f'{username}_tweets.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(tweets_data, f, indent=4, default=str)
    print(f"Detailed data saved to {json_filename}")

    # Print some statistics
    print(f"\nTweet Statistics for @{username}:")
    print(f"Total tweets: {len(tweets_data)}")
    if tweets_data:
        print(f"Most liked tweet: {df.sort_values(by='favorite_count', ascending=False).iloc[0]['favorite_count']} likes")
        print(f"Average likes: {df['favorite_count'].mean():.1f}")

    # Preview top tweets
    print(f"\nTop 3 most liked tweets:")
    top_tweets = df.sort_values(by='favorite_count', ascending=False).head(3)
    for i, (_, tweet) in enumerate(top_tweets.iterrows(), 1):
        print(f"{i}. ({tweet['favorite_count']} likes) {tweet['full_text'][:100]}...")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python twitter_scraper.py <username_or_url> [max_tweets]")
        print("Examples:")
        print("  python twitter_scraper.py jiaseedpudding")
        print("  python twitter_scraper.py https://x.com/jiaseedpudding")
        print("  python twitter_scraper.py jiaseedpudding 30")
        sys.exit(1)

    username_or_url = sys.argv[1]
    max_tweets = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # Extract username from URL or use as-is
    username = extract_username_from_url(username_or_url)

    print(f"Starting Twitter scraper for @{username}")
    print(f"Max tweets to scrape: {max_tweets}")

    # Scrape tweets
    tweets_data = await scrape_user_tweets(username, max_tweets)

    if tweets_data:
        # Append to golden_labels.txt in required format
        append_to_golden_labels(username, tweets_data)

        # Save detailed data for analysis
        save_detailed_data(username, tweets_data)

        print(f"\n✅ Successfully scraped {len(tweets_data)} tweets from @{username}")
    else:
        print(f"\n❌ No tweets found for @{username}")
        # Still add to golden_labels.txt with empty list
        append_to_golden_labels(username, [])

if __name__ == "__main__":
    asyncio.run(main())