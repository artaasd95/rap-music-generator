# Style-Based Rap Generation System Documentation

## Overview
The style-based rap generation system allows users to generate rap lyrics in the style of specific artists or with particular stylistic characteristics. The system uses a combination of DPO (Direct Preference Optimization) training and style-based generation to create lyrics that match the desired style.

## Components

### 1. Style Data Preparation (`prepare_style_data.py`)
This script prepares the training data for style-based generation by analyzing rap lyrics and extracting style characteristics.

#### Key Functions:

##### `extract_style_reference(artist_name, lyrics)`
Extracts style characteristics from lyrics and artist information.

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
The system automatically adds these characteristics based on lyrical content:
- Extensive vocabulary (if lyrics contain more than 100 words)
- Emotional depth (if lyrics contain emotional words)
- Social commentary (if lyrics contain social commentary words)

##### `prepare_style_dataset()`
Processes the rap lyrics dataset and creates a style-based training dataset.

**Output:**
- Creates a JSON file at `data/rap_style_dataset.json` containing processed lyrics with style references

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

1. Run the style data preparation script:
```bash
python prepare_style_data.py
```

2. The script will:
   - Load the rap lyrics dataset
   - Process each song to extract style characteristics
   - Create a JSON file with processed data
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

2. **Prompts:**
   - Use 2-3 lines as a prompt
   - Match the style of the prompt with the desired output style
   - Include context or theme in the prompt

3. **Generation Parameters:**
   - Adjust `max_length` based on desired output length
   - Use `num_return_sequences` to get multiple variations
   - Experiment with different prompts for the same style

## Limitations

1. The style analysis is based on predefined characteristics and simple word matching
2. The system may not capture subtle nuances of each artist's style
3. Generated lyrics may not perfectly match the original artist's style
4. Style references are limited to the predefined set of artists

## Future Improvements

1. Enhanced style analysis using NLP techniques
2. More sophisticated style characteristic extraction
3. Support for more artists and styles
4. Better handling of regional and cultural styles
5. Integration with audio generation for complete rap production 