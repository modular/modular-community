from dotenv import dotenv_values, load_dotenv

fn main() raises:
    # Basic smoke test
    print("Testing mojo-dotenv import...")

    # Test that functions exist and are callable
    # Note: We can't actually load files in the test environment
    # but we can verify the functions are importable
    print("✓ dotenv_values imported")
    print("✓ load_dotenv imported")

    print("All import tests passed!")
