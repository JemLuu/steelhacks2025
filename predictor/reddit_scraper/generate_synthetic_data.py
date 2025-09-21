#!/usr/bin/env python3
"""
Synthetic Mental Health Data Generator

This script uses Claude to generate synthetic social media posts along with
their mental health condition scores for training data augmentation.

Generates 4200 synthetic posts with scores for:
- depression, anxiety, ptsd, schizophrenia, bipolar, eating_disorder, adhd, overall_score
"""

import json
import os
import sys
import time
from typing import Dict, List, Any
import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv

class SyntheticDataGenerator:
    """Generate synthetic mental health posts with scores using Claude."""

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

    def create_generation_prompt(self, batch_size: int, focus_condition: str = None) -> str:
        """Create a prompt for Claude to generate synthetic posts with scores."""

        focus_instruction = ""
        if focus_condition:
            focus_instruction = f"\nFOCUS: Generate posts that primarily relate to {focus_condition}, but include variety in severity and presentation."

        prompt = f"""You are a mental health data generator creating realistic synthetic social media posts for training an AI model. Generate {batch_size} diverse, authentic-sounding social media posts that represent various mental health experiences.

{focus_instruction}

REQUIREMENTS:
1. Create posts that sound like real people sharing their experiences
2. Include the FULL range of severity levels (0.0 to 1.0 across different conditions)
3. Use natural, varied language patterns and emotional expressions
4. Include posts from different perspectives (struggling, recovering, seeking help, supporting others, completely healthy)
5. Make posts realistic but not overly clinical or dramatic
6. Vary post length from short (1 sentence) to longer (3-4 sentences)
7. Include different contexts (work, relationships, daily life, therapy, etc.)
8. Generate posts with 0.0 scores (completely neutral/healthy mental state)
9. Generate posts with 1.0 scores (severe mental health crises)
10. Include everyday posts that show no mental health concerns

MENTAL HEALTH CONDITIONS TO SCORE (0.0 to 1.0):
- depression: Sadness, hopelessness, low energy, loss of interest
- anxiety: Worry, fear, panic, social anxiety, physical symptoms
- ptsd: Trauma responses, flashbacks, avoidance, hypervigilance
- schizophrenia: Delusions, hallucinations, disorganized thinking
- bipolar: Mood swings, mania, depression cycles
- eating_disorder: Body image issues, food relationships, weight concerns
- adhd: Attention issues, hyperactivity, impulsivity, executive function
- overall_score: General mental health distress level

EXAMPLES OF VARIETY TO INCLUDE:

NEUTRAL/HEALTHY (0.0-0.2 scores):
- "Had a great day at work, feeling productive and happy"
- "Just finished a wonderful dinner with friends, life is good"
- "Excited for my vacation next week, been planning this for months"
- "Love my new hobby, pottery class is so relaxing"

MILD (0.3-0.4 scores):
- "Feeling a bit stressed about the presentation tomorrow"
- "Having trouble sleeping lately, probably too much coffee"

MODERATE (0.5-0.6 scores):
- "Just had my first therapy session, feeling hopeful for the first time in months"
- "Can't focus on anything today, my brain feels like it's in a million places"
- "Three days without a panic attack, small wins matter"

SEVERE (0.7-1.0 scores):
- "The voices are getting louder again, need to call my doctor"
- "Ate normally for the first time in weeks, recovery is hard but possible"
- "Manic episode hit hard last night, spent $500 on stuff I don't need"
- "Flashbacks from the accident keep interrupting my work meetings"
- "Can't get out of bed, everything feels impossible right now"

Generate EXACTLY {batch_size} posts. For each post, provide:
1. The social media post text (realistic, natural language)
2. Scores for all 8 mental health indicators

Respond ONLY with a valid JSON array containing {batch_size} objects:
[
{{"text": "realistic social media post here", "depression": 0.0, "anxiety": 0.0, "ptsd": 0.0, "schizophrenia": 0.0, "bipolar": 0.0, "eating_disorder": 0.0, "adhd": 0.0, "overall_score": 0.0}},
{{"text": "another realistic post", "depression": 0.0, "anxiety": 0.0, "ptsd": 0.0, "schizophrenia": 0.0, "bipolar": 0.0, "eating_disorder": 0.0, "adhd": 0.0, "overall_score": 0.0}},
...
]

Use decimal values between 0.0 and 1.0. Make scores realistic and consistent with the post content.

IMPORTANT: Include a balanced distribution:
- 30% neutral/healthy posts (0.0-0.2 scores)
- 30% mild concerns (0.3-0.4 scores)
- 25% moderate issues (0.5-0.6 scores)
- 15% severe cases (0.7-1.0 scores)

Ensure variety across all mental health conditions and include everyday social media posts that show no mental health concerns."""

        return prompt

    def generate_batch(self, batch_size: int, focus_condition: str = None, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Generate a batch of synthetic posts with scores."""
        prompt = self.create_generation_prompt(batch_size, focus_condition)

        for attempt in range(max_retries):
            try:
                print(f"Generating batch of {batch_size} posts (attempt {attempt + 1}/{max_retries})...")
                print(f"  → Sending request to Claude Sonnet 4...")

                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=12000,  # Reduced for 100 posts
                    temperature=0.67,  # Lower temperature for better JSON format compliance
                    messages=[{"role": "user", "content": prompt}]
                )

                print(f"  → Received response, parsing JSON...")
                response_text = response.content[0].text.strip()
                print(f"  → Response length: {len(response_text)} characters")

                generated_data = json.loads(response_text)
                print(f"  → Successfully parsed JSON with {len(generated_data)} items")

                # Validate the generated data
                print(f"  → Validating {len(generated_data)} generated items...")
                if len(generated_data) == batch_size:
                    validated_data = []
                    required_fields = ["text"] + self.conditions + ["overall_score"]

                    for i, item in enumerate(generated_data):
                        if all(field in item for field in required_fields):
                            # Ensure scores are between 0.0 and 1.0
                            for field in self.conditions + ["overall_score"]:
                                item[field] = max(0.0, min(1.0, float(item[field])))
                            validated_data.append(item)
                        else:
                            print(f"  ⚠️  Warning: Missing fields in generated item {i+1}")

                    if len(validated_data) == batch_size:
                        print(f"  ✅ Successfully validated {len(validated_data)} posts")
                        return validated_data
                    else:
                        print(f"  ⚠️  Warning: Only {len(validated_data)}/{batch_size} posts were valid")

                else:
                    print(f"  ⚠️  Warning: Expected {batch_size} posts, got {len(generated_data)}")

            except json.JSONDecodeError as e:
                print(f"  ❌ JSON parsing failed on attempt {attempt + 1}: {e}")
                print(f"  📄 Response preview: {response_text[:200]}...")

            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait_time = 300 + (attempt * 60)  # 5+ minutes with increasing wait
                    print(f"  🕒 Rate limit hit. Waiting {wait_time} seconds ({wait_time/60:.1f} minutes)...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        print(f"  ⏳ Waiting 5 seconds before retry...")
                        time.sleep(5)  # Minimal wait on errors

        print("  ❌ Failed to generate valid batch after all retries")
        return []

    def append_batch_to_file(self, batch_data: List[Dict[str, Any]], filename: str = "post_data2.py", is_first_batch: bool = False):
        """Append a batch of generated posts to the output file."""
        try:
            mode = 'w' if is_first_batch else 'a'
            with open(filename, mode, encoding='utf-8') as f:
                if is_first_batch:
                    # Write file header
                    f.write("# Synthetic Mental Health Reddit Posts Dataset\n")
                    f.write("# Generated automatically using Claude Sonnet 4\n")
                    f.write("# Each post is scored for various mental health conditions (0.0 - 1.0)\n\n")
                    f.write("post_data = [\n")

                # Write batch data
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

        except IOError as e:
            print(f"Error writing batch to file '{filename}': {e}")

    def finalize_file(self, filename: str = "post_data2.py"):
        """Close the post_data array in the file."""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.seek(f.tell() - 2)  # Remove last comma
                f.write("\n]\n")
        except IOError as e:
            print(f"Error finalizing file '{filename}': {e}")

    def generate_all_data(self, total_posts: int = 1200, batch_size: int = 100, filename: str = "post_data2.py"):
        """Generate all synthetic data in batches."""
        generated_posts = []
        total_batches = (total_posts + batch_size - 1) // batch_size

        print(f"Generating {total_posts} synthetic posts in batches of {batch_size}...")
        print(f"Results will be saved incrementally to '{filename}'")

        # Define focus conditions to ensure variety
        focus_conditions = [
            None,  # General mix
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd"
        ]

        for batch_num in range(1, total_batches + 1):
            # Calculate actual batch size for this batch
            current_batch_size = min(batch_size, total_posts - len(generated_posts))

            # Rotate through focus conditions to ensure variety
            focus_condition = focus_conditions[(batch_num - 1) % len(focus_conditions)]

            print(f"\nGenerating batch {batch_num}/{total_batches} (posts {len(generated_posts) + 1}-{len(generated_posts) + current_batch_size})...")
            if focus_condition:
                print(f"Focus condition: {focus_condition}")

            # Generate the batch
            batch_data = self.generate_batch(current_batch_size, focus_condition)

            if batch_data:
                print(f"  💾 Saving {len(batch_data)} posts to file...")
                # Append to file immediately
                self.append_batch_to_file(batch_data, filename, is_first_batch=(batch_num == 1))
                generated_posts.extend(batch_data)

                print(f"✅ Completed {len(generated_posts)}/{total_posts} posts ({len(generated_posts)/total_posts*100:.1f}%)")
                print(f"📁 Batch {batch_num} saved to file")
            else:
                print(f"❌ Failed to generate batch {batch_num}, skipping...")

            # No rate limiting for speed (only wait on rate limit errors)
            # time.sleep removed for maximum speed

        # Finalize the file
        self.finalize_file(filename)
        print(f"\nCompleted generating {len(generated_posts)} synthetic posts!")
        print(f"Results saved to '{filename}'")
        return generated_posts

def main():
    """Generate 4200 synthetic mental health posts."""
    print("Synthetic Mental Health Data Generator")
    print("=" * 40)

    generator = SyntheticDataGenerator()

    # Ask for confirmation
    try:
        response = input("Generate 1200 synthetic posts? This will use API credits. (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    except EOFError:
        print("Running non-interactively, proceeding with generation...")

    # Generate all data
    generator.generate_all_data(total_posts=1200, batch_size=100)

    print("\n" + "=" * 40)
    print("Generation complete!")
    print("You can now import this data with: from post_data2 import post_data")

if __name__ == "__main__":
    main()