#!/usr/bin/env python3
"""
Utility script to analyze saved memory evolution data.
"""

import pickle
import json
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append('src')
from test_advanced import ConversationMemoryTracker, MemoryEvolutionSnapshot

def analyze_memory_evolution(pickle_path: str):
    """Analyze memory evolution data from a pickle file."""
    
    # Load the data
    with open(pickle_path, 'rb') as f:
        memory_evolution_data = pickle.load(f)
    
    print(f"Loaded memory evolution data for {len(memory_evolution_data)} conversations")
    print("="*60)
    
    # Analyze each conversation
    for i, tracker in enumerate(memory_evolution_data):
        print(f"\nConversation {i+1} (ID: {tracker.conversation_id})")
        print(f"Sample ID: {tracker.sample_id}")
        print(f"Total snapshots: {len(tracker.snapshots)}")
        
        # Get summary
        summary = tracker.get_memory_evolution_summary()
        print(f"Summary keys: {list(summary.keys())}")
        print(f"Total created: {summary.get('total_created', 0)}")
        print(f"Total evolved: {summary.get('total_evolutions', 0)}")
        print(f"Unique memories: {summary.get('unique_memories', 0)}")
        print(f"Evolution actions: {summary.get('evolution_actions', {})}")
        
        # Show some sample snapshots
        if tracker.snapshots:
            print(f"\nFirst few memory snapshots:")
            for j, snapshot in enumerate(tracker.snapshots[:3]):
                print(f"  Snapshot {j+1}:")
                print(f"    Turn {snapshot.turn_index} - {snapshot.speaker}")
                print(f"    Action: {snapshot.evolution_action}")
                print(f"    Memory: {snapshot.memory_content[:100]}...")
                print(f"    Keywords: {snapshot.memory_keywords}")
        
        print("-" * 40)
    
    # Overall statistics
    total_memories = sum(len(tracker.snapshots) for tracker in memory_evolution_data)
    total_created = sum(sum(1 for snap in tracker.snapshots if snap.evolution_action == 'created') 
                       for tracker in memory_evolution_data)
    total_evolved = sum(sum(1 for snap in tracker.snapshots if snap.evolution_action == 'evolved') 
                       for tracker in memory_evolution_data)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total conversations: {len(memory_evolution_data)}")
    print(f"Total memory snapshots: {total_memories}")
    print(f"Total memories created: {total_created}")
    print(f"Total memories evolved: {total_evolved}")
    print(f"Average snapshots per conversation: {total_memories / len(memory_evolution_data):.2f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_memory_evolution.py <pickle_file_path>")
        print("\nExample:")
        print("  python analyze_memory_evolution.py output_memory_evolution.pkl")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    
    if not os.path.exists(pickle_path):
        print(f"Error: File {pickle_path} does not exist")
        sys.exit(1)
    
    try:
        analyze_memory_evolution(pickle_path)
    except Exception as e:
        print(f"Error analyzing memory evolution data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
