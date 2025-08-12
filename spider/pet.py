import time
import pandas as pd
from pytrends.request import TrendReq


def get_pet_popularity(pet_names, timeframe="today 5-y"):
    """
    Query Google Trends for multiple pet types and compare their popularity.

    Args:
        pet_names: List of pet names (without 'pet' prefix)
        timeframe: Time period to analyze

    Returns:
        Dictionary of normalized popularity scores
    """
    # Add "pet" prefix to each pet name
    keywords = [f"pet {pet}" for pet in pet_names]

    # Need to query in batches of 5 (Google Trends limit)
    batch_size = 5
    all_results = {}
    reference_scores = {}
    reference_keyword = None

    # Process in batches
    for i in range(0, len(keywords), batch_size):
        batch_keywords = keywords[i : i + batch_size]

        # For batches after the first one, include a reference keyword from previous batch
        if i > 0 and reference_keyword:
            # Use the middle-scoring keyword from previous batch as reference
            batch_keywords = [reference_keyword] + batch_keywords[: batch_size - 1]

        print(f"Querying batch: {', '.join(batch_keywords)}")

        # Query Google Trends
        pytrends = TrendReq(hl="en-US", tz=360)
        try:
            pytrends.build_payload(batch_keywords, cat=0, timeframe=timeframe)
            batch_results = pytrends.interest_over_time()

            # If results are empty, wait and try again
            if batch_results.empty:
                print("Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                pytrends = TrendReq(hl="en-US", tz=360)
                pytrends.build_payload(batch_keywords, cat=0, timeframe=timeframe)
                batch_results = pytrends.interest_over_time()

            if not batch_results.empty:
                # Remove isPartial column if exists
                if "isPartial" in batch_results.columns:
                    batch_results = batch_results.drop("isPartial", axis=1)

                # Calculate average scores for this batch
                batch_means = {}
                for keyword in batch_keywords:
                    if keyword in batch_results.columns:
                        batch_means[keyword] = batch_results[keyword].mean()

                # For the first batch, just store the scores
                if i == 0:
                    for keyword, score in batch_means.items():
                        all_results[keyword] = score

                    # Choose reference keyword for next batch (middle-scoring item)
                    sorted_scores = sorted(batch_means.items(), key=lambda x: x[1])
                    middle_idx = len(sorted_scores) // 2
                    reference_keyword = sorted_scores[middle_idx][0]
                    reference_scores[reference_keyword] = sorted_scores[middle_idx][1]

                # For subsequent batches, normalize based on the reference keyword
                else:
                    # Get scaling factor from reference keyword
                    if (
                        reference_keyword in batch_means
                        and batch_means[reference_keyword] > 0
                    ):
                        scaling_factor = (
                            reference_scores[reference_keyword]
                            / batch_means[reference_keyword]
                        )

                        # Apply scaling to all keywords except the reference
                        for keyword in batch_keywords:
                            if keyword != reference_keyword and keyword in batch_means:
                                all_results[keyword] = (
                                    batch_means[keyword] * scaling_factor
                                )

                    # Choose new reference keyword for next batch
                    new_batch_keywords = [
                        k for k in batch_keywords if k != reference_keyword
                    ]
                    if new_batch_keywords:
                        # Get scores for current batch (excluding reference)
                        current_scores = {
                            k: batch_means[k]
                            for k in new_batch_keywords
                            if k in batch_means
                        }
                        if current_scores:
                            sorted_scores = sorted(
                                current_scores.items(), key=lambda x: x[1]
                            )
                            middle_idx = len(sorted_scores) // 2
                            reference_keyword = sorted_scores[middle_idx][0]
                            reference_scores[reference_keyword] = (
                                sorted_scores[middle_idx][1] * scaling_factor
                            )

            # Wait to avoid hitting rate limits
            time.sleep(10)

        except Exception as e:
            print(f"Error querying trends: {e}")
            continue

    # If no results were found
    if not all_results:
        print("Failed to retrieve data from Google Trends")
        return {}

    # Final normalization - set highest to 1.0
    max_score = max(all_results.values()) if all_results else 1
    normalized_scores = {pet: score / max_score for pet, score in all_results.items()}

    return normalized_scores


def display_normalized_scores(scores, pet_names):
    """
    Display normalized scores in a readable format.

    Args:
        scores: Dictionary of normalized scores
        pet_names: Original pet names list
    """
    print("\nPet Popularity (normalized, highest = 1.0):")
    print("-" * 40)

    # Map scores back to original pet names
    pet_scores = {}
    for pet in pet_names:
        key = f"pet {pet}"
        if key in scores:
            pet_scores[pet] = scores[key]

    # Sort by popularity (descending)
    sorted_pets = sorted(pet_scores.items(), key=lambda x: x[1], reverse=True)

    # Display each pet with its score
    for pet, score in sorted_pets:
        # Create a simple bar visualization
        bar = "█" * int(score * 80)
        print(f"{pet:10}: {score:.4f} {bar}")


def main():
    # List of pets to compare
    pet_names = [
        "dog",
        "turtle",
        "cat",
        "hamster",
        "horse",
        "fish",
        "bird",
        "rabbit",
        "snake",
        "lizardparrot",
        "pig",
        "duck",
        "cow",
        "chicken",
        "goat",
        "sheep",
        "ferret",
        "chinchilla",
        "fox",
        "spider",
        "mantis",
    ]

    # Time period to analyze
    timeframe = "today 12-m"  # Last 12 months

    print(f"Comparing popularity of pets: {', '.join(pet_names)}")
    print(f"Timeframe: {timeframe}")

    # Get normalized popularity scores
    scores = get_pet_popularity(pet_names, timeframe)

    # Display results
    if scores:
        display_normalized_scores(scores, pet_names)
    else:
        print("Could not retrieve popularity scores.")


if __name__ == "__main__":
    main()
