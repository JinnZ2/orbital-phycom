import requests
import json
import os

def harvest_tle_catalog():
    """
    Scrapes CelesTrak for 'Active' or 'Debris' elements.
    Saves to local Atlas for offline processing.
    """
    # Focusing on 10cm+ debris (salvageable 'Shapes')
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
    
    print("Scraping CelesTrak for Orbital Resources...")
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            catalog = response.json()
            with open("atlas/debris_catalog.json", "w") as f:
                json.dump(catalog, f, indent=2)
            print(f"✓ Harvested {len(catalog)} Potential Resource Reservoirs.")
        else:
            print(f"Institutional Friction: HTTP {response.status_code}")
    except Exception as e:
        print(f"Anxiety: Prediction Error during scrape: {e}")

# Usage: Run this once when the 'Needs-Flow' (internet) is high.
# harvest_tle_catalog()
