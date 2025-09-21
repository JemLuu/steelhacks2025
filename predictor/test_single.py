#!/usr/bin/env python3
"""
Simple single text tester for Mental Health API
Quick and easy way to test individual texts
"""

import requests
import json
import sys

# API Configuration
API_URL = "https://jluu196--mental-health-api-fastapi-app.modal.run"

def analyze_text(text: str):
    """Analyze a single text and display results"""

    # Make API request
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text}
    )

    if response.status_code != 200:
        print(f"❌ Error: API returned status {response.status_code}")
        return None

    result = response.json()

    # Display results
    print("\n" + "="*50)
    print("🧠 MENTAL HEALTH ANALYSIS")
    print("="*50)
    print(f"\nInput Text:")
    print(f"'{text}'\n")
    print("-"*50)
    print("Condition Scores:")
    print("-"*50)

    # Create visual representation
    conditions = [
        ("Depression", result['depression']),
        ("Anxiety", result['anxiety']),
        ("PTSD", result['ptsd']),
        ("Schizophrenia", result['schizophrenia']),
        ("Bipolar", result['bipolar']),
        ("Eating Disorder", result['eating_disorder']),
        ("ADHD", result['adhd'])
    ]

    # Show scores with bar charts
    for condition, score in conditions:
        # Create bar (max 30 characters wide)
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)

        # Color coding
        if score >= 0.7:
            indicator = "🔴"  # High
        elif score >= 0.4:
            indicator = "🟡"  # Medium
        else:
            indicator = "🟢"  # Low

        print(f"{indicator} {condition:15} [{bar}] {score:.2f}")

    print("-"*50)
    print(f"📊 Overall Score:   {result['overall_score']:.2f}")
    print("="*50)

    # Interpretation
    print("\nInterpretation:")
    high_conditions = [c[0] for c in conditions if c[1] >= 0.7]
    moderate_conditions = [c[0] for c in conditions if 0.4 <= c[1] < 0.7]

    if high_conditions:
        print(f"⚠️  High indicators: {', '.join(high_conditions)}")
    if moderate_conditions:
        print(f"⚡ Moderate indicators: {', '.join(moderate_conditions)}")
    if not high_conditions and not moderate_conditions:
        print("✅ No significant mental health indicators detected")

    print("\n" + "="*50)

    # Return raw scores
    return result

def main():
    """Main function - handles command line arguments or interactive mode"""

    print("\n🧠 Mental Health API - Single Text Analyzer")
    print(f"API: {API_URL}")

    # Check if text was provided as command line argument
    if len(sys.argv) > 1:
        # Join all arguments as the text to analyze
        text = " ".join(sys.argv[1:])
        print(f"\nAnalyzing provided text...")
        analyze_text(text)
    else:
        # Interactive mode
        print("\nEnter text to analyze (or 'quit' to exit):")
        print("Examples:")
        print("  - I feel anxious and worried all the time")
        print("  - Everything seems hopeless lately")
        print("  - I'm so happy and excited about life!")

        while True:
            print("\n" + "-"*50)
            text = input("Enter text: ").strip()

            if text.lower() in ['quit', 'exit', 'q', '']:
                print("👋 Goodbye!")
                break

            try:
                analyze_text(text)

                # Ask if user wants to continue
                again = input("\nAnalyze another text? (y/n): ").lower()
                if again not in ['y', 'yes']:
                    print("👋 Goodbye!")
                    break

            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()