def get_class_from_response(response_string):
    """Extract class name from LLM response string."""
    try:
        # Split by 'assistant<|end_header_id|>\n\n' and take the last part
        parts = response_string.split('assistant<|end_header_id|>\n\n')
        if len(parts) < 2:
            print("No assistant response found")
            return None
            
        # Take the part after the split and clean it up
        response = parts[1].split('<|eot_id|>')[0].strip().rstrip('.')
        print(f"Found response: '{response}'")
        return response
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    print("Testing with example string...")
    
    test_input = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What object is shown in this image? Choose one from these options: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Respond with only the class name.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\ndog.<|eot_id|>'
    
    result = get_class_from_response(test_input)
    print(f"Extracted class: {result}")
