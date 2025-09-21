#!/usr/bin/env python3
"""
Resume Reddit Post Scoring Script

This script resumes scoring from a specific batch to avoid losing progress.
"""

import json
import os
import sys
import time
from typing import Dict, List, Any
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

class ResumeScorer:
    """Resume scoring Reddit posts from a specific batch."""

    def __init__(self):
        """Initialize the Claude API client."""
        load_dotenv()
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not found.")
            sys.exit(1)

        self.client = Anthropic(api_key=api_key)
        self.conditions = [
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd"
        ]

    def load_posts(self, filename: str = "to_label.txt") -> List[Dict[str, Any]]:
        """Load and parse Reddit posts from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        posts = []
        for subreddit, post_list in data.items():
            for post_text in post_list:
                cleaned_text = post_text.replace(" | ", ". ")
                posts.append({
                    "text": cleaned_text,
                    "source_subreddit": subreddit
                })
        return posts

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

    def score_batch(self, posts_batch: List[Dict[str, Any]], max_retries: int = 5) -> List[Dict[str, float]]:
        """Score a batch of posts with longer waits for rate limits."""
        prompt = self.create_batch_scoring_prompt(posts_batch)

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} for batch...")
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.content[0].text.strip()
                scores_array = json.loads(response_text)

                # Handle extra results gracefully
                validated_scores = []
                required_fields = self.conditions + ["overall_score"]
                scores_to_process = min(len(scores_array), len(posts_batch))

                for i in range(scores_to_process):
                    scores = scores_array[i]
                    if all(field in scores for field in required_fields):
                        for field in required_fields:
                            scores[field] = max(0.0, min(1.0, float(scores[field])))
                        validated_scores.append(scores)
                    else:
                        print(f"Warning: Missing fields in response for post {i+1}")
                        validated_scores.append({condition: 0.5 for condition in required_fields})

                # Fill missing scores if needed
                while len(validated_scores) < len(posts_batch):
                    validated_scores.append({condition: 0.5 for condition in required_fields})

                return validated_scores

            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = 300 + (attempt * 60)  # 5+ minutes with increasing wait
                    print(f"Rate limit hit. Waiting {wait_time} seconds ({wait_time/60:.1f} minutes)...")
                    time.sleep(wait_time)
                else:
                    print(f"Error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(30)

        print("Failed to score batch after all retries")
        return [{condition: 0.5 for condition in self.conditions + ["overall_score"]} for _ in posts_batch]

    def append_batch_to_file(self, batch_data: List[Dict[str, Any]], filename: str = "post_data.py"):
        """Append a batch to the existing file."""
        with open(filename, 'a', encoding='utf-8') as f:
            for post in batch_data:
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

    def replace_batch_in_file(self, batch_data: List[Dict[str, Any]], batch_num: int, filename: str = "post_data.py"):
        """Replace a specific batch in the file by rewriting the end."""
        # Read current file
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find where to cut off (before the batch we want to replace)
        posts_before_batch = (batch_num - 1) * 50
        # Each post takes about 11 lines in the file
        lines_per_post = 11
        cut_line = 4 + (posts_before_batch * lines_per_post)  # 4 header lines + posts

        # Keep everything before the batch we're replacing
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(lines[:cut_line])

        # Append the new batch
        self.append_batch_to_file(batch_data, filename)

    def finalize_file(self, filename: str = "post_data.py"):
        """Close the post_data array in the file."""
        with open(filename, 'a', encoding='utf-8') as f:
            f.seek(f.tell() - 2)  # Remove last comma
            f.write("\n]\n")

def main():
    """Resume scoring from batch 8."""
    print("Resume Reddit Post Scoring")
    print("=" * 30)

    scorer = ResumeScorer()
    posts = scorer.load_posts("to_label.txt")

    # Define which batches to redo
    start_batch = 8  # Batch 8 (posts 351-400) had default scores
    end_batch = 16   # Continue to the end
    batch_size = 50

    print(f"Resuming from batch {start_batch} (replacing default scores)")
    print("Starting immediately...")

    for batch_num in range(start_batch, end_batch + 1):
        batch_start = (batch_num - 1) * batch_size
        batch_end = min(batch_start + batch_size, len(posts))
        batch = posts[batch_start:batch_end]

        print(f"\nScoring batch {batch_num}/{end_batch} (posts {batch_start + 1}-{batch_end})...")

        # Score the batch
        batch_scores = scorer.score_batch(batch)

        # Create batch data
        batch_data = []
        for post, scores in zip(batch, batch_scores):
            scored_post = {"text": post['text'], **scores}
            batch_data.append(scored_post)

        # Replace this batch in the file
        if batch_num == start_batch:
            scorer.replace_batch_in_file(batch_data, batch_num)
        else:
            scorer.append_batch_to_file(batch_data)

        print(f"Batch {batch_num} completed and saved")

        # Wait between batches
        if batch_num < end_batch:
            print("Waiting 30 seconds before next batch...")
            time.sleep(30)

    # Finalize the file
    scorer.finalize_file()
    print(f"\nCompleted! All batches {start_batch}-{end_batch} processed")

if __name__ == "__main__":
    main()