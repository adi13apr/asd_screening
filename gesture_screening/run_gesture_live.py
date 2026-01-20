from gesture_screening.gesture_module import run_gesture_screening

if __name__ == "__main__":
    print("Running gesture screening (10 seconds)...")
    output = run_gesture_screening(duration_sec=10)
    print("\nGesture Screening Output:\n")
    print(output)
