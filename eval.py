def calculate_text_metrics(predicted_text, ground_truth):
    """
    Calculate accuracy, precision, recall and F1 score between predicted text and ground truth
    at both character and word levels
    
    Args:
        predicted_text (str): The predicted or generated text
        ground_truth (str): The reference or ground truth text
        
    Returns:
        dict: Dictionary containing metrics at character and word levels
    """
    # Character-level metrics
    # Convert strings to lists of characters
    pred_chars = list(predicted_text)
    true_chars = list(ground_truth)
    
    # Word-level metrics
    # Split strings into words
    pred_words = predicted_text.split()
    true_words = ground_truth.split()
    
    # Character-level evaluation
    char_tp = sum(1 for c in pred_chars if c in true_chars)
    char_fp = len(pred_chars) - char_tp
    char_fn = len(true_chars) - char_tp
    
    # Word-level evaluation
    word_tp = sum(1 for w in pred_words if w in true_words)
    word_fp = len(pred_words) - word_tp
    word_fn = len(true_words) - word_tp
    
    # Calculate metrics for character level
    try:
        char_precision = char_tp / (char_tp + char_fp) if (char_tp + char_fp) > 0 else 0
        char_recall = char_tp / (char_tp + char_fn) if (char_tp + char_fn) > 0 else 0
        char_f1 = 2 * (char_precision * char_recall) / (char_precision + char_recall) if (char_precision + char_recall) > 0 else 0
        
        # For character accuracy, consider position (exact match)
        correct_chars = sum(1 for p, t in zip(pred_chars, true_chars) if p == t)
        total_chars = max(len(pred_chars), len(true_chars))
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    except ZeroDivisionError:
        char_precision, char_recall, char_f1, char_accuracy = 0, 0, 0, 0
    
    # Calculate metrics for word level
    try:
        word_precision = word_tp / (word_tp + word_fp) if (word_tp + word_fp) > 0 else 0
        word_recall = word_tp / (word_tp + word_fn) if (word_tp + word_fn) > 0 else 0
        word_f1 = 2 * (word_precision * word_recall) / (word_precision + word_recall) if (word_precision + word_recall) > 0 else 0
        
        # For word accuracy, consider position (exact match)
        correct_words = sum(1 for p, t in zip(pred_words, true_words) if p == t)
        total_words = max(len(pred_words), len(true_words))
        word_accuracy = correct_words / total_words if total_words > 0 else 0
    except ZeroDivisionError:
        word_precision, word_recall, word_f1, word_accuracy = 0, 0, 0, 0
    
    return {
        "character_level": {
            "accuracy": char_accuracy,
            "precision": char_precision,
            "recall": char_recall,
            "f1_score": char_f1
        },
        "word_level": {
            "accuracy": word_accuracy,
            "precision": word_precision,
            "recall": word_recall,
            "f1_score": word_f1
        }
    }


pred = "আমার বিশ্ববিদ্যালয়ের নাম বাংলাদেশইউনিভার্সিটি অব বিজনেস এন্ড টেকনোলজিএটি ঢাকার মিরপুরে অবস্থিত এখানে শিক্ষার্থীআনুমানিক দশ হাজার অধ্যয়নরত আছে। বিশ্ববিদ্যালয়টিরাংলাদেশের শীর্ষস্থানীয় বিশ্ববিদ্যালয়গুলোর একটি। আমি আমার বিশ্ববিদ্যালয় সম্পর্কে গর্বিত"

truth = "আমার বিশ্ববিদ্যালয়ের নাম বাংলাদেশ ইউনিভার্সিটি অব বিজনেস এন্ড টেকনোলজি এটি ঢাকার মিরপুরে অবস্থিত এখানে আনুমানিক দশ হাজার শিক্ষার্থী অধ্যয়নরত আছে বিশ্ববিদ্যালয়টি বাংলাদেশের শীর্ষস্থানীয় বিশ্ববিদ্যালয়গুলোর একটি আমি আমার বিশ্ববিদ্যালয় সম্পর্কে গর্বিত"

# metrics = calculate_text_metrics(pred, truth)
# print(metrics)

from PIL import Image, ImageEnhance

def process_image(input_image):
    """
    Converts the input image to grayscale and increases its contrast.

    Args:
        input_image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The processed image (grayscale and contrast-enhanced).
    """
    # Convert image to grayscale
    grayscale_image = input_image.convert("L")

    # Increase contrast
    enhancer = ImageEnhance.Contrast(grayscale_image)
    enhanced_image = enhancer.enhance(2.0)  # Increase contrast by a factor (e.g., 2.0)

    return enhanced_image

img = Image.open("testImage.jpg")
processed_i = process_image(img)
processed_i.save("process.jpg")
processed_i.show()