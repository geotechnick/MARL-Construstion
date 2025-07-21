#!/usr/bin/env python3
"""
Quick demo of the site builder functionality.
"""

def demo_text_builder():
    """Demo the text-based builder with a preset."""
    print("=== Text-Based Site Builder Demo ===")
    
    from examples.text_site_builder import TextSiteBuilder
    
    # Create a 12x12 site
    builder = TextSiteBuilder(12, 12)
    
    # Create a coordination scenario
    print("Creating coordination scenario...")
    builder.create_preset("coordination")
    
    # Show the result
    builder.print_site()
    
    # Save it
    builder.save_site("demo_site.json")
    print("\nSaved as demo_site.json")
    
    print("\nYou can now:")
    print("  1. Edit with: python examples/text_site_builder.py --file demo_site.json")
    print("  2. Train on it: python examples/train_custom_site.py --site demo_site.json --demo")

def demo_gui_builder():
    """Demo the GUI builder creation."""
    print("=== GUI Site Builder Demo ===")
    
    try:
        from marl_construction.envs import ConstructionSiteBuilder
        import pygame
        
        # Create a small site
        builder = ConstructionSiteBuilder(8, 8)
        
        # Add some elements programmatically
        builder.add_material(2, 2, "brick")
        builder.add_material(5, 5, "steel")
        builder.add_task(6, 6)
        builder.add_spawn(1, 1)
        builder.add_spawn(7, 7)
        
        # Save the demo site
        builder.save_site()
        
        # Clean up pygame
        if pygame.get_init():
            pygame.quit()
            
        print("Created demo site with GUI builder")
        print("Saved as custom_site_8x8.json")
        print("\nTo use the full GUI:")
        print("  python examples/site_builder.py")
        
    except Exception as e:
        print(f"GUI builder demo failed: {e}")
        print("Use text builder instead")

def main():
    print("Site Builder Demo")
    print("=" * 40)
    
    # Demo text builder
    demo_text_builder()
    print()
    
    # Demo GUI builder  
    demo_gui_builder()
    print()
    
    print("Site Builder Features:")
    print("  * Interactive design of construction scenarios")
    print("  * Multiple material types and agent roles")
    print("  * Save/load custom scenarios")
    print("  * Live testing with MARL agents")
    print("  * Both GUI and text-based interfaces")

if __name__ == "__main__":
    main()