#!/usr/bin/env python3
"""
Ground-Atmospheric Coupling Demonstration

Shows how natural ground thermal sources create detectable atmospheric
signatures through pure physics coupling. No artificial intervention needed.

Run time: ~4 minutes
Output: Comprehensive visualization of ground-atmosphere energy flow
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from atmospheric.ground_calibration import demonstration_experiment

if __name__ == '__main__':
    print()
    print("GROUND-ATMOSPHERIC COUPLING DEMONSTRATION")
    print()
    print("This simulation demonstrates:")
    print("  • Natural thermal sources (solar, thermal mass, evaporation)")
    print("  • Ground-to-atmosphere energy coupling")
    print("  • Multi-station detection network")
    print("  • Communication through coordinated natural processes")
    print()
    print("All within natural thermal ranges - no artificial intervention")
    print()
    input("Press Enter to begin simulation...")
    print()
    
    demonstration_experiment()
