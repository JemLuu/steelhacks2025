# Reddit Top Posts Scraper

A production-ready Python script that scrapes the top 100 posts from a subreddit's "top this year" section using web scraping (no API credentials required).

## Features

- Fetches top posts from any public subreddit for the current year
- Extracts both post titles and text content
- No Reddit API credentials needed - uses Reddit's JSON API
- Outputs results in JSON format to `to_label.txt` with format: "title | text"
- **Automatically appends to existing file** - scrape multiple subreddits into one file
- Comprehensive error handling for edge cases
- Command-line interface with argument validation
- Progress indicators during scraping
- Handles private subreddits, banned subreddits, and network issues
- Automatic retry logic with exponential backoff
- Filters out stickied posts and advertisements

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

That's it! No API setup required.

## Usage

### Basic Usage

```bash
python scraper.py <subreddit_name>
```

### Examples

```bash
# Scrape r/programming
python scraper.py programming

# Scrape r/MachineLearning
python scraper.py MachineLearning

# Scrape r/python
python scraper.py python

# Scrape multiple subreddits into the same file
python scraper.py depression       # Creates to_label.txt with depression posts
python scraper.py anxiety          # Appends anxiety posts to existing file
python scraper.py mentalhealth     # Appends mentalhealth posts to existing file
```

### Advanced Options

```bash
# Scrape only 50 posts instead of 100
python scraper.py programming --limit 50

# Save to a custom filename
python scraper.py programming --output programming_posts.json

# Combine options
python scraper.py python --limit 75 --output python_top_posts.json
```

### Command Line Arguments

- `subreddit`: (Required) Name of the subreddit to scrape (without r/ prefix)
- `--limit`: Number of posts to fetch (default: 100, max: 1000)
- `--output`: Output filename (default: to_label.txt)

## Output Format

The script saves results in JSON format with title and text content separated by " | ". When scraping multiple subreddits, they are all stored in the same file:

```json
{
  "depression": [
    "Post title 1 | Post text content here...",
    "Post title 2 | Another post's text content...",
    ...
  ],
  "anxiety": [
    "Different post title | Different text content...",
    "Another title | More text content...",
    ...
  ],
  "programming": [
    "Tech post title | Tech post content...",
    "Code question | [no text content]",
    ...
  ]
}
```

## Appending Data

The script automatically appends to existing files:
- If `to_label.txt` exists, new subreddit data is added to it
- If a subreddit already exists in the file, its data will be replaced with the new scrape
- To start fresh, simply delete or rename the existing file

## Error Handling

The script handles various error conditions:

- **Invalid subreddit names**: Validates and cleans subreddit names
- **Non-existent subreddits**: Provides clear error messages
- **Private subreddits**: Detects and reports access issues
- **Banned subreddits**: Handles forbidden access gracefully
- **Network issues**: Retries and reports connection problems
- **Blocked requests**: Uses browser-like headers to avoid detection

## Rate Limits

This scraper is respectful to Reddit's servers:
- Includes 1-second delays between page requests
- Uses automatic retry logic with exponential backoff
- The script includes progress indicators for long requests
- Mimics browser behavior to avoid being blocked

## Troubleshooting

### Common Issues

1. **"Subreddit not found"**
   - Check the subreddit name spelling
   - Make sure the subreddit exists and is public

2. **"Access denied"**
   - The subreddit may be private or banned
   - Try a different public subreddit

3. **"Network error"**
   - Check your internet connection
   - Reddit servers may be temporarily unavailable
   - The script will automatically retry failed requests

4. **"No posts found"**
   - The subreddit may not have many posts from this year
   - Try a more active subreddit

### Getting Help

If you encounter issues:
1. Verify the subreddit name is correct and the subreddit is public
2. Try with a well-known subreddit like "programming" or "python"
3. Check your internet connection

## Security Notes

- This script uses web scraping and doesn't require any credentials
- The script is read-only and cannot post or modify content
- No user authentication is required

## License

This project is released under the MIT License.