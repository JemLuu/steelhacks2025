#!/usr/bin/env python3
"""
Mental Health Post Scoring Script

This script reads Reddit posts from to_label.txt and uses Claude Sonnet 4
to score each post for various mental health conditions, then outputs
a post_data.py file with the scored data.

Requirements:
    - anthropic library (pip install anthropic)
    - python-dotenv library (pip install python-dotenv)
    - ANTHROPIC_API_KEY in .env file or environment variable
"""

import json
import os
import sys
import time
from typing import Dict, List, Any
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

class MentalHealthScorer:
    """Scores Reddit posts for mental health conditions using Claude API."""

    def __init__(self):
        """Initialize the Claude API client."""
        # Load environment variables from .env file
        load_dotenv()

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found.")
            print("Please set your Anthropic API key in one of these ways:")
            print("1. Create a .env file with: ANTHROPIC_API_KEY=your_api_key_here")
            print("2. Set environment variable: export ANTHROPIC_API_KEY='your_api_key_here'")
            sys.exit(1)

        self.client = Anthropic(api_key=api_key)

        # Mental health conditions to score
        self.conditions = [
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd"
        ]

    def load_posts(self, filename: str = "to_label.txt") -> List[Dict[str, Any]]:
        """Load and parse Reddit posts from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            posts = []
            for subreddit, post_list in data.items():
                for post_text in post_list:
                    # Replace " | " with ". " as specified
                    cleaned_text = post_text.replace(" | ", ". ")
                    posts.append({
                        "text": cleaned_text,
                        "source_subreddit": subreddit
                    })

            print(f"Loaded {len(posts)} posts from {len(data)} subreddits")
            return posts

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file '{filename}'.")
            sys.exit(1)

    def create_batch_scoring_prompt(self, posts_batch: List[Dict[str, Any]]) -> str:
        """Create a prompt for Claude to score multiple posts at once."""
        posts_text = ""
        for i, post in enumerate(posts_batch, 1):
            posts_text += f"POST {i}:\nText: \"{post['text']}\"\nSource: r/{post['source_subreddit']}\n\n"

        prompt = f"""You are a mental health professional analyzing social media posts. Please analyze the following {len(posts_batch)} Reddit posts and provide scores for various mental health conditions.

{posts_text}

TASK: Score each post for indicators of the following mental health conditions on a scale of 0.0 to 1.0:
- depression (0.0 = no indicators, 1.0 = strong indicators)
- anxiety (0.0 = no indicators, 1.0 = strong indicators)
- ptsd (0.0 = no indicators, 1.0 = strong indicators)
- schizophrenia (0.0 = no indicators, 1.0 = strong indicators)
- bipolar (0.0 = no indicators, 1.0 = strong indicators)
- eating_disorder (0.0 = no indicators, 1.0 = strong indicators)
- adhd (0.0 = no indicators, 1.0 = strong indicators)
- overall_score (0.0 = good mental health, 1.0 = severe mental health concerns)

Consider:
- Language patterns and emotional tone
- Specific symptoms or behaviors mentioned
- Severity of distress expressed
- The source subreddit context
- Self-reported experiences and feelings

Respond ONLY with a valid JSON array containing {len(posts_batch)} objects, one for each post in order:
[
{{"depression": 0.0, "anxiety": 0.0, "ptsd": 0.0, "schizophrenia": 0.0, "bipolar": 0.0, "eating_disorder": 0.0, "adhd": 0.0, "overall_score": 0.0}},
{{"depression": 0.0, "anxiety": 0.0, "ptsd": 0.0, "schizophrenia": 0.0, "bipolar": 0.0, "eating_disorder": 0.0, "adhd": 0.0, "overall_score": 0.0}},
...
]

Use decimal values between 0.0 and 1.0. Do not include any other text or explanation."""

        return prompt

    def score_batch(self, posts_batch: List[Dict[str, Any]], max_retries: int = 3) -> List[Dict[str, float]]:
        """Score a batch of posts using Claude API with retry logic."""
        prompt = self.create_batch_scoring_prompt(posts_batch)

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",  # Latest Claude Sonnet 4
                    max_tokens=8000,  # Increased for batch processing
                    temperature=0.1,  # Low temperature for consistent scoring
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                # Parse the JSON response
                response_text = response.content[0].text.strip()

                # Try to extract JSON array from response
                try:
                    scores_array = json.loads(response_text)

                    # Handle cases where we get more or fewer responses than expected
                    validated_scores = []
                    required_fields = self.conditions + ["overall_score"]

                    # Take only the number of scores we need (handle extra results)
                    scores_to_process = min(len(scores_array), len(posts_batch))

                    for i in range(scores_to_process):
                        scores = scores_array[i]
                        if all(field in scores for field in required_fields):
                            # Ensure all values are between 0.0 and 1.0
                            for field in required_fields:
                                scores[field] = max(0.0, min(1.0, float(scores[field])))
                            validated_scores.append(scores)
                        else:
                            print(f"Warning: Missing fields in response for post {i+1}")
                            validated_scores.append({condition: 0.5 for condition in required_fields})

                    # If we got fewer responses than expected, fill with defaults
                    while len(validated_scores) < len(posts_batch):
                        print(f"Warning: Got fewer responses than expected, using default for post {len(validated_scores) + 1}")
                        validated_scores.append({condition: 0.5 for condition in required_fields})

                    if len(scores_array) != len(posts_batch):
                        print(f"Note: Expected {len(posts_batch)} scores, got {len(scores_array)}, proceeding anyway")

                    return validated_scores

                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON response on attempt {attempt + 1}")
                    print(f"Response: {response_text[:500]}...")

                # Wait before retry
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Longer exponential backoff
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait_time = (2 ** attempt) * 10  # Much longer wait for rate limits
                    print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        time.sleep(wait_time)

        # If all attempts failed, return default scores for all posts
        print(f"Warning: Failed to score batch after {max_retries} attempts. Using default scores.")
        default_scores = {condition: 0.5 for condition in self.conditions + ["overall_score"]}
        return [default_scores for _ in posts_batch]

    def append_batch_to_file(self, batch_data: List[Dict[str, Any]], filename: str = "post_data.py", is_first_batch: bool = False):
        """Append a batch of scored posts to the output file."""
        try:
            mode = 'w' if is_first_batch else 'a'
            with open(filename, mode, encoding='utf-8') as f:
                if is_first_batch:
                    # Write file header
                    f.write("# Mental Health Reddit Posts Dataset\n")
                    f.write("# Generated automatically from Reddit data using Claude Sonnet 4\n")
                    f.write("# Each post is scored for various mental health conditions (0.0 - 1.0)\n\n")
                    f.write("post_data = [\n")

                # Write batch data
                for i, post in enumerate(batch_data):
                    f.write("    {\n")
                    f.write(f'        "text": {repr(post["text"])},\n')
                    f.write(f'        "depression": {post["depression"]},\n')
                    f.write(f'        "anxiety": {post["anxiety"]},\n')
                    f.write(f'        "ptsd": {post["ptsd"]},\n')
                    f.write(f'        "schizophrenia": {post["schizophrenia"]},\n')
                    f.write(f'        "bipolar": {post["bipolar"]},\n')
                    f.write(f'        "eating_disorder": {post["eating_disorder"]},\n')
                    f.write(f'        "adhd": {post["adhd"]},\n')
                    f.write(f'        "overall_score": {post["overall_score"]}\n')
                    f.write("    },\n")

        except IOError as e:
            print(f"Error writing batch to file '{filename}': {e}")

    def finalize_file(self, filename: str = "post_data.py"):
        """Close the post_data array in the file."""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                # Remove the last comma and close the array
                f.seek(f.tell() - 2)  # Go back 2 characters to overwrite ",\n"
                f.write("\n]\n")
        except IOError as e:
            print(f"Error finalizing file '{filename}': {e}")

    def score_all_posts(self, posts: List[Dict[str, Any]], filename: str = "post_data.py") -> List[Dict[str, Any]]:
        """Score all posts in batches and append results incrementally."""
        scored_posts = []
        total_posts = len(posts)
        batch_size = 50

        print(f"Starting to score {total_posts} posts in batches of {batch_size}...")
        print(f"Results will be saved incrementally to '{filename}'")

        # Process posts in batches
        for batch_start in range(0, total_posts, batch_size):
            batch_end = min(batch_start + batch_size, total_posts)
            batch = posts[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total_posts + batch_size - 1) // batch_size

            print(f"Scoring batch {batch_num}/{total_batches} (posts {batch_start + 1}-{batch_end})...")

            # Score the batch
            batch_scores = self.score_batch(batch)

            # Create the final data structures
            batch_data = []
            for i, (post, scores) in enumerate(zip(batch, batch_scores)):
                scored_post = {
                    "text": post['text'],
                    **scores
                }
                scored_posts.append(scored_post)
                batch_data.append(scored_post)

            # Append this batch to the file immediately
            self.append_batch_to_file(batch_data, filename, is_first_batch=(batch_num == 1))

            # Progress indicator
            completed = len(scored_posts)
            print(f"Completed {completed}/{total_posts} posts ({completed/total_posts*100:.1f}%)")
            print(f"Batch {batch_num} saved to file")

            # Rate limiting - increased wait time to avoid 429 errors
            if batch_num < total_batches:  # Don't sleep after the last batch
                print("Waiting 10 seconds before next batch...")
                time.sleep(10)

        # Finalize the file
        self.finalize_file(filename)
        print(f"Completed scoring all {total_posts} posts!")
        print(f"Results saved to '{filename}'")
        return scored_posts

    def write_post_data_file(self, scored_posts: List[Dict[str, Any]], filename: str = "post_data.py"):
        """Write the scored posts to a Python file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Mental Health Reddit Posts Dataset\n")
                f.write("# Generated automatically from Reddit data using Claude Sonnet 4\n")
                f.write("# Each post is scored for various mental health conditions (0.0 - 1.0)\n\n")

                f.write("post_data = [\n")

                for i, post in enumerate(scored_posts):
                    # Format the post data nicely
                    f.write("    {\n")
                    f.write(f'        "text": {repr(post["text"])},\n')
                    f.write(f'        "depression": {post["depression"]},\n')
                    f.write(f'        "anxiety": {post["anxiety"]},\n')
                    f.write(f'        "ptsd": {post["ptsd"]},\n')
                    f.write(f'        "schizophrenia": {post["schizophrenia"]},\n')
                    f.write(f'        "bipolar": {post["bipolar"]},\n')
                    f.write(f'        "eating_disorder": {post["eating_disorder"]},\n')
                    f.write(f'        "adhd": {post["adhd"]},\n')
                    f.write(f'        "overall_score": {post["overall_score"]}\n')

                    if i < len(scored_posts) - 1:
                        f.write("    },\n")
                    else:
                        f.write("    }\n")

                f.write("]\n")

            print(f"Successfully wrote {len(scored_posts)} scored posts to '{filename}'")

        except IOError as e:
            print(f"Error writing to file '{filename}': {e}")
            sys.exit(1)

def main():
    """Main function to run the scoring process."""
    print("Mental Health Post Scoring Script")
    print("=" * 40)

    # Initialize the scorer
    scorer = MentalHealthScorer()

    # Load posts from file
    posts = scorer.load_posts("to_label.txt")

    # Ask for confirmation before proceeding (this will use API credits)
    print(f"\nThis will score {len(posts)} posts using Claude API.")
    print("This may take a while and will use API credits.")

    try:
        response = input("Do you want to proceed? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    except EOFError:
        # Running non-interactively, proceed automatically
        print("Running non-interactively, proceeding with scoring...")
        pass

    # Score all posts (results are written incrementally)
    scored_posts = scorer.score_all_posts(posts)

    print("\n" + "=" * 40)
    print("Scoring complete!")
    print(f"Generated post_data.py with {len(scored_posts)} scored posts")
    print("You can now import this data with: from post_data import post_data")

if __name__ == "__main__":
    main()