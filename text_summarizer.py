from transformers import pipeline

def main():
    # Initialize the summarization pipeline
    # This will download the default model (distilbart-cnn-12-6) upon first run if not already cached.
    print("Loading the model... (This may take a moment on the first run)")
    summarizer = pipeline("summarization")

    # The input text
    text = """
    Artificial Intelligence is transforming many industries by allowing machines to perform tasks that normally require human intelligence. It is widely used in healthcare, finance, robotics, and automation.
    """

    print("-" * 50)
    print("Original Text:")
    print(text.strip())
    print("-" * 50)

    # Calculate input length to avoid 'max_length > input_length' warnings
    input_word_count = len(text.split())
    # We set max_length safely below the input word count
    safe_max_length = min(50, max(10, int(input_word_count * 0.8)))
    safe_min_length = min(20, max(5, int(input_word_count * 0.2)))

    # Generate the summary
    print("Generating summary...")
    summary = summarizer(text, max_length=safe_max_length, min_length=safe_min_length, do_sample=False)
    
    # Extract and display the summary
    print("-" * 50)
    print("Summary:")
    print(summary[0]['summary_text'])
    print("-" * 50)

if __name__ == "__main__":
    main()
