# Style-Based Rap Generation System Documentation

## Overview
The style-based rap generation system allows users to generate rap lyrics in the style of specific artists or with particular stylistic characteristics. The system uses advanced NLP techniques, rhyme analysis, and flow pattern detection to create comprehensive style references for rap lyrics generation.

## Components

### 1. Style Data Preparation (`prepare_style_data.py`)
This script prepares the training data for style-based generation by performing advanced analysis of rap lyrics and extracting detailed style characteristics.

#### Key Components:

##### `StyleAnalyzer` Class
A comprehensive class for analyzing rap lyrics using advanced NLP techniques.

**Methods:**

###### `analyze_rhyme_scheme(lyrics: str) -> dict`
Analyzes the rhyme patterns in the lyrics using the CMU pronunciation dictionary.

**Returns:**
- `rhyme_density`: Ratio of rhyming lines to total lines
- `common_rhymes`: Most frequent rhyme patterns
- `rhyme_variety`: Diversity of rhyme schemes used

###### `analyze_flow(lyrics: str) -> dict`
Analyzes the flow characteristics of the lyrics.

**Returns:**
- `avg_word_length`: Average length of words
- `word_length_variety`: Standard deviation of word lengths
- `fast_density`: Ratio of short words (1-3 letters)
- `complex_density`: Ratio of long words (8+ letters)
- `repetitive_density`: Ratio of repeated word patterns

###### `analyze_content(lyrics: str) -> dict`
Analyzes the content and themes of the lyrics.

**Returns:**
- `sentiment`: Polarity and subjectivity scores
- `theme_scores`: Scores for different themes:
  - Violence
  - Wealth
  - Success
  - Struggle
  - Social commentary
- `vocabulary_richness`: Ratio of unique words to total words

##### `extract_style_reference(artist_name: str, lyrics: str) -> str`
Extracts style characteristics from lyrics and artist information using advanced analysis.

**Parameters:**
- `artist_name` (str): Name of the artist
- `lyrics` (str): The lyrics to analyze

**Returns:**
- `str`: A style reference string containing the artist's style characteristics

**Supported Artists and Their Base Styles:**
- Eminem: Fast-paced, complex wordplay, aggressive tone, and dark humor
- Kendrick Lamar: Conscious lyrics, complex storytelling, and social commentary
- Jay-Z: Business-oriented, confident, and sophisticated wordplay
- Nas: Street poetry, vivid imagery, and introspective storytelling
- Tupac: Emotional depth, social consciousness, and raw authenticity
- Kanye West: Innovative, self-reflective, and boundary-pushing
- Drake: Melodic flow, emotional vulnerability, and pop sensibilities
- J. Cole: Thoughtful, introspective, and socially conscious
- Travis Scott: Atmospheric, psychedelic, and modern trap influence
- Lil Wayne: Creative wordplay, punchlines, and unique metaphors

**Dynamic Style Characteristics:**
The system automatically detects these characteristics based on advanced analysis:
- Fast-paced delivery (based on word length patterns)
- Complex wordplay (based on vocabulary analysis)
- Varied flow patterns (based on word length variety)
- Dense rhyming (based on rhyme scheme analysis)
- Diverse rhyme schemes (based on rhyme variety)
- Extensive vocabulary (based on vocabulary richness)
- Personal expression (based on sentiment subjectivity)
- Social commentary (based on theme analysis)
- Emotional depth (based on theme and sentiment analysis)

##### `prepare_style_dataset() -> None`
Processes the rap lyrics dataset and creates a comprehensive style-based training dataset.

**Output:**
Creates a JSON file at `data/rap_style_dataset.json` containing:
```json
{
    "artist": "artist_name",
    "lyrics": "original_lyrics",
    "style_reference": "extracted_style_reference",
    "style_analysis": {
        "rhyme_analysis": {
            "rhyme_density": float,
            "common_rhymes": list,
            "rhyme_variety": float
        },
        "flow_analysis": {
            "avg_word_length": float,
            "word_length_variety": float,
            "fast_density": float,
            "complex_density": float,
            "repetitive_density": float
        },
        "content_analysis": {
            "sentiment": {
                "polarity": float,
                "subjectivity": float
            },
            "theme_scores": {
                "violence": float,
                "wealth": float,
                "success": float,
                "struggle": float,
                "social": float
            },
            "vocabulary_richness": float
        }
    }
}
```

### 2. Style-Based Generation (`style_based_generation.ipynb`)

#### Key Classes:

##### `StyleBasedGenerator`
Main class for generating style-based rap lyrics.

**Initialization:**
```python
generator = StyleBasedGenerator(
    model_path='checkpoints/dpo_trained_model',
    tokenizer_path='checkpoints/dpo_trained_model'
)
```

**Methods:**

###### `generate_lyrics(style_reference, prompt, max_length=200, num_return_sequences=1)`
Generates rap lyrics in the specified style.

**Parameters:**
- `style_reference` (str): Description of the desired style
- `prompt` (str): Starting lyrics or context
- `max_length` (int): Maximum length of generated lyrics
- `num_return_sequences` (int): Number of different versions to generate

**Returns:**
- `list`: List of generated lyrics strings

## Usage Examples

### 1. Basic Style-Based Generation
```python
from style_based_generation import StyleBasedGenerator

# Initialize the generator
generator = StyleBasedGenerator(
    model_path='checkpoints/dpo_trained_model',
    tokenizer_path='checkpoints/dpo_trained_model'
)

# Generate lyrics in Eminem's style
style_reference = "Eminem's style: Fast-paced, complex wordplay, aggressive tone, and dark humor"
prompt = "Look, I was gonna go easy on you not to hurt your feelings\nBut I'm only going to get this one chance"

generated_lyrics = generator.generate_lyrics(
    style_reference=style_reference,
    prompt=prompt,
    max_length=200,
    num_return_sequences=3
)
```

### 2. Custom Style Generation
```python
# Generate lyrics with custom style characteristics
custom_style = "Fast-paced, aggressive, with extensive vocabulary and emotional depth"
prompt = "In the streets where the weak don't survive"

generated_lyrics = generator.generate_lyrics(
    style_reference=custom_style,
    prompt=prompt,
    max_length=150,
    num_return_sequences=2
)
```

## Training Data Preparation

1. Install required dependencies:
```bash
pip install nltk textblob numpy
```

2. Run the style data preparation script:
```bash
python prepare_style_data.py
```

3. The script will:
   - Load the rap lyrics dataset
   - Process each song using advanced style analysis
   - Create a JSON file with processed data including:
     - Artist name
     - Original lyrics
     - Style reference
     - Detailed style analysis (rhyme, flow, content)
   - Save the dataset to `data/rap_style_dataset.json`

## Style Reference Format

Style references should follow this format:
```
"[Artist]'s style: [Base characteristics], with [Additional characteristics]"
```

Example:
```
"Eminem's style: Fast-paced, complex wordplay, aggressive tone, and dark humor, with extensive vocabulary and emotional depth"
```

## Best Practices

1. **Style References:**
   - Be specific about the style characteristics
   - Include both technical and emotional aspects
   - Mention any unique features of the artist's style
   - Consider the advanced analysis metrics when describing styles

2. **Prompts:**
   - Use 2-3 lines as a prompt
   - Match the style of the prompt with the desired output style
   - Include context or theme in the prompt
   - Consider the flow patterns in your prompt

3. **Generation Parameters:**
   - Adjust `max_length` based on desired output length
   - Use `num_return_sequences` to get multiple variations
   - Experiment with different prompts for the same style
   - Consider the style analysis metrics when evaluating output

## Limitations

1. The style analysis is based on statistical patterns and may not capture all nuances
2. The system may not perfectly match the original artist's style
3. Style references are limited to the predefined set of artists
4. Advanced analysis requires significant computational resources
5. Some complex rhyme schemes may not be fully detected

## Future Improvements

1. Enhanced style analysis using deep learning techniques
2. More sophisticated rhyme scheme detection
3. Support for more artists and styles
4. Better handling of regional and cultural styles
5. Integration with audio generation for complete rap production
6. Real-time style analysis and feedback
7. Custom style reference creation tools
8. Style transfer between different artists 