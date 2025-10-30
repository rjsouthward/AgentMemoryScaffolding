#!/usr/bin/env python3
"""
Test memory evolution tracking functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from test_advanced import ConversationMemoryTracker, MemoryEvolutionSnapshot, save_memory_evolution_data, load_memory_evolution_data

def test_memory_evolution_tracking():
    """Test the memory evolution tracking functionality."""
    print("Testing memory evolution tracking...")
    
    # Create a test tracker
    tracker = ConversationMemoryTracker(
        sample_id=0,
        conversation_id="test_conversation_1",
        snapshots=[],
        memory_timeline={}
    )
    
    # Add some test snapshots
    snapshots_data = [
        ("2023-01-01 10:00:00", 1, "user", "I love programming", "mem_1", "Programming is enjoyable", ["programming", "enjoyment"], ["hobby"], "User expresses enjoyment of programming", 0.8, 1, "created"),
        ("2023-01-01 10:01:00", 2, "assistant", "What languages do you use?", "mem_2", "User asks about programming languages", ["programming", "languages"], ["question"], "Discussion about programming languages", 0.7, 1, "created"),
        ("2023-01-01 10:02:00", 3, "user", "I mainly use Python and JavaScript", "mem_1", "Programming is enjoyable, especially Python and JavaScript", ["programming", "enjoyment", "python", "javascript"], ["hobby", "languages"], "User enjoys programming, specifically Python and JavaScript", 0.9, 2, "evolved"),
    ]
    
    for timestamp, turn_index, speaker, content, memory_id, memory_content, keywords, tags, context, importance, retrieval_count, action in snapshots_data:
        snapshot = MemoryEvolutionSnapshot(
            timestamp=timestamp,
            turn_index=turn_index,
            speaker=speaker,
            content=content,
            memory_id=memory_id,
            memory_content=memory_content,
            memory_keywords=keywords,
            memory_tags=tags,
            memory_context=context,
            importance_score=importance,
            retrieval_count=retrieval_count,
            evolution_action=action,
            related_memories=[]
        )
        tracker.add_snapshot(snapshot)
    
    # Test summary generation
    summary = tracker.get_memory_evolution_summary()
    print(f"Summary: {summary}")
    
    # Test saving and loading
    test_data = [tracker]
    save_memory_evolution_data(test_data, "./test_memory_evolution")
    
    # Load and verify
    loaded_data = load_memory_evolution_data("./test_memory_evolution/memory_evolution_complete.pkl")
    print(f"Loaded {len(loaded_data)} trackers")
    print(f"First tracker has {len(loaded_data[0].snapshots)} snapshots")
    
    print("✅ Memory evolution tracking test completed successfully!")

if __name__ == "__main__":
    test_memory_evolution_tracking()
