from visual_attention.behavior_module import run_behavior_screening

if __name__ == "__main__":
    print("Running CV behavioral analysis...")
    output = run_behavior_screening(10)
    print("\nBehavior Screening Output:\n")
    print(output)
