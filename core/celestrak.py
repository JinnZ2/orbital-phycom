"""
CelesTrak TLE catalog harvester.

Downloads active satellite TLE data from CelesTrak for use in
orbital analysis and debris tracking.

Usage:
    python -m core.celestrak                     # Save to default path
    python -m core.celestrak --output catalog.json  # Custom output
"""

import json
import os
import argparse

import requests


CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"


def harvest_tle_catalog(output_path="atlas/debris_catalog.json"):
    """
    Download active satellite TLE catalog from CelesTrak.

    Args:
        output_path: Path to save the JSON catalog.

    Returns:
        List of catalog entries, or empty list on failure.
    """
    print(f"Fetching TLE catalog from CelesTrak...")
    try:
        response = requests.get(CELESTRAK_URL, timeout=15)
        response.raise_for_status()
        catalog = response.json()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(catalog, f, indent=2)
        print(f"Harvested {len(catalog)} entries -> {output_path}")
        return catalog
    except requests.RequestException as e:
        print(f"Failed to fetch catalog: {e}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CelesTrak TLE catalog")
    parser.add_argument("--output", default="atlas/debris_catalog.json",
                        help="Output file path")
    args = parser.parse_args()
    harvest_tle_catalog(args.output)
