#!/usr/bin/env python3
"""
Test script for the site builder functionality.
"""

def test_text_builder():
    """Test the text-based site builder."""
    print("=== Testing Text-Based Site Builder ===")
    
    from examples.text_site_builder import TextSiteBuilder
    
    # Create builder
    builder = TextSiteBuilder(10, 10)
    
    # Test adding elements
    builder.add_material(2, 3, "brick")
    builder.add_material(5, 5, "steel")
    builder.add_task(7, 8)
    builder.add_spawn(1, 1)
    builder.add_spawn(9, 9)
    
    # Test display
    builder.print_site()
    
    # Test save/load
    builder.save_site("test_site.json")
    
    # Clear and reload
    builder.clear_site()
    builder.load_site("test_site.json")
    
    print("Text builder test completed!")
    
def test_gui_builder_creation():
    """Test GUI builder creation without running the main loop."""
    print("=== Testing GUI Builder Creation ===")
    
    try:
        from marl_construction.envs import ConstructionSiteBuilder
        
        # Just test creation, don't run
        builder = ConstructionSiteBuilder(8, 8)
        
        # Test some methods
        builder.add_material(2, 3, "brick")
        builder.add_task(5, 5)
        builder.add_spawn(1, 1)
        
        # Test site conversion
        site = builder.to_construction_site()
        
        print(f"Created {site.width}x{site.height} site")
        print(f"Materials: {len(site.materials)}")
        print(f"Tasks: {len(site.tasks)}")
        
        # Close pygame properly
        import pygame
        if pygame.get_init():
            pygame.quit()
            
        print("GUI builder creation test completed!")
        return True
        
    except Exception as e:
        print(f"GUI builder test failed: {e}")
        return False

def main():
    print("Testing Site Builder Components...")
    print("=" * 50)
    
    # Test text builder
    test_text_builder()
    print()
    
    # Test GUI builder creation
    gui_success = test_gui_builder_creation()
    print()
    
    if gui_success:
        print("✓ Both site builders working!")
        print("\nUsage:")
        print("  GUI Builder: python examples/site_builder.py")
        print("  Text Builder: python examples/text_site_builder.py")
    else:
        print("! GUI builder has issues, use text builder:")
        print("  python examples/text_site_builder.py")

if __name__ == "__main__":
    main()