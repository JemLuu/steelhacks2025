import json
import random
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import pandas as pd

class MentalHealthDataProcessor:
    """Advanced data processing for mental health classification"""

    def __init__(self, data_path: str = None):
        self.conditions = [
            "depression", "anxiety", "ptsd", "schizophrenia",
            "bipolar", "eating_disorder", "adhd"
        ]
        self.data = None
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str):
        """Load Reddit post data"""
        with open(data_path, 'r') as f:
            content = f.read()

        # Execute the Python file to get post_data
        local_vars = {}
        exec(content, {}, local_vars)
        self.data = local_vars['post_data']
        print(f"Loaded {len(self.data)} samples")

    def analyze_data_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of scores across conditions"""
        if not self.data:
            raise ValueError("No data loaded")

        analysis = {}
        df = pd.DataFrame(self.data)

        for condition in self.conditions + ["overall_score"]:
            scores = df[condition].values
            analysis[condition] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "zero_count": int(np.sum(scores == 0.0)),
                "high_score_count": int(np.sum(scores >= 0.7)),
            }

        # Text length analysis
        text_lengths = [len(sample['text']) for sample in self.data]
        analysis["text_stats"] = {
            "mean_length": float(np.mean(text_lengths)),
            "max_length": int(np.max(text_lengths)),
            "min_length": int(np.min(text_lengths)),
        }

        return analysis

    def format_for_training(self,
                          train_split: float = 0.8,
                          val_split: float = 0.1,
                          test_split: float = 0.1,
                          instruction_variants: bool = True) -> Dict[str, List[Dict]]:
        """Format data for instruction fine-tuning with multiple variants"""

        if not self.data:
            raise ValueError("No data loaded")

        # Create instruction variants for better generalization
        instruction_templates = [
            "Analyze the following text for mental health indicators and return a JSON object with scores (0.0-1.0) for each condition:\n\n{text}",
            "Please evaluate this text for mental health conditions and provide scores from 0.0 to 1.0 for each category:\n\n{text}",
            "Given the following text, assess the likelihood of various mental health conditions (scale 0.0-1.0):\n\n{text}",
            "Rate this text for mental health indicators on a scale of 0.0 to 1.0 for each condition:\n\n{text}",
            "Analyze this text and provide mental health condition scores (0.0-1.0):\n\n{text}",
        ]

        formatted_samples = []

        for sample in self.data:
            # Create target JSON
            target = {
                "depression": sample['depression'],
                "anxiety": sample['anxiety'],
                "ptsd": sample['ptsd'],
                "schizophrenia": sample['schizophrenia'],
                "bipolar": sample['bipolar'],
                "eating_disorder": sample['eating_disorder'],
                "adhd": sample['adhd'],
                "overall_score": sample['overall_score']
            }

            if instruction_variants:
                # Use random instruction template for each sample
                template = random.choice(instruction_templates)
            else:
                template = instruction_templates[0]

            # Format as conversation
            conversation = f"""<bos><start_of_turn>user
{template.format(text=sample['text'])}<end_of_turn>
<start_of_turn>model
{json.dumps(target, indent=2)}<end_of_turn><eos>"""

            formatted_samples.append({
                "text": conversation,
                "input_text": sample['text'],
                "target_scores": target,
                "raw_sample": sample
            })

        # Split the data
        total_samples = len(formatted_samples)
        train_size = int(total_samples * train_split)
        val_size = int(total_samples * val_split)

        # Shuffle data
        random.shuffle(formatted_samples)

        train_data = formatted_samples[:train_size]
        val_data = formatted_samples[train_size:train_size + val_size]
        test_data = formatted_samples[train_size + val_size:]

        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
            "stats": {
                "total_samples": total_samples,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "test_samples": len(test_data),
            }
        }

    def create_evaluation_prompts(self, samples: List[Dict]) -> List[str]:
        """Create evaluation prompts for model testing"""
        prompts = []

        for sample in samples:
            prompt = f"""<bos><start_of_turn>user
Analyze the following text for mental health indicators and return a JSON object with scores (0.0-1.0) for each condition:

{sample['input_text']}<end_of_turn>
<start_of_turn>model
"""
            prompts.append(prompt)

        return prompts

    def evaluate_predictions(self,
                           true_scores: List[Dict],
                           predicted_scores: List[Dict]) -> Dict[str, float]:
        """Evaluate model predictions against ground truth"""

        metrics = {}

        for condition in self.conditions + ["overall_score"]:
            true_vals = [sample[condition] for sample in true_scores]
            pred_vals = [sample.get(condition, 0.0) for sample in predicted_scores]

            # Regression metrics
            mse = mean_squared_error(true_vals, pred_vals)
            mae = mean_absolute_error(true_vals, pred_vals)

            # Classification metrics (threshold at 0.5)
            true_binary = [1 if x >= 0.5 else 0 for x in true_vals]
            pred_binary = [1 if x >= 0.5 else 0 for x in pred_vals]

            accuracy = accuracy_score(true_binary, pred_binary)

            metrics[condition] = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(np.sqrt(mse)),
                "accuracy": float(accuracy),
            }

        # Overall metrics
        all_true = []
        all_pred = []

        for condition in self.conditions:
            all_true.extend([sample[condition] for sample in true_scores])
            all_pred.extend([sample.get(condition, 0.0) for sample in predicted_scores])

        metrics["overall"] = {
            "mse": float(mean_squared_error(all_true, all_pred)),
            "mae": float(mean_absolute_error(all_true, all_pred)),
            "rmse": float(np.sqrt(mean_squared_error(all_true, all_pred))),
        }

        return metrics

    def save_processed_data(self, data_splits: Dict, output_path: str):
        """Save processed data to JSON"""

        # Convert to serializable format
        serializable_data = {}
        for split_name, split_data in data_splits.items():
            if split_name == "stats":
                serializable_data[split_name] = split_data
            else:
                serializable_data[split_name] = [
                    {
                        "text": sample["text"],
                        "input_text": sample["input_text"],
                        "target_scores": sample["target_scores"]
                    }
                    for sample in split_data
                ]

        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    processor = MentalHealthDataProcessor("reddit_scraper/post_data.py")

    # Analyze data distribution
    analysis = processor.analyze_data_distribution()
    print("Data Analysis:")
    for condition, stats in analysis.items():
        if condition != "text_stats":
            print(f"{condition}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, high_scores={stats['high_score_count']}")

    print(f"Text stats: avg_length={analysis['text_stats']['mean_length']:.0f}")

    # Process data for training
    data_splits = processor.format_for_training()
    print(f"\nData splits: {data_splits['stats']}")

    # Save processed data
    processor.save_processed_data(data_splits, "processed_mental_health_data.json")