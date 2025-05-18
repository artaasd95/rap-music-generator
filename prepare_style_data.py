import json
import os
from datasets import load_dataset
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
import re
from collections import Counter
import numpy as np
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt')
nltk.download('cmudict')

class StyleAnalyzer:
    """Advanced style analysis for rap lyrics."""
    
    def __init__(self):
        self.cmudict = cmudict.dict()
        self.flow_patterns = {
            'fast': r'\b\w{1,3}\b',  # Short words
            'complex': r'\b\w{8,}\b',  # Long words
            'repetitive': r'(\b\w+\b)(?:\s+\1\b)+',  # Repeated words
        }
        
    def analyze_rhyme_scheme(self, lyrics: str) -> dict:
        """Analyze the rhyme scheme of the lyrics.
        
        Args:
            lyrics (str): The lyrics to analyze
            
        Returns:
            dict: Rhyme scheme characteristics
        """
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        rhyme_scheme = []
        
        for line in lines:
            words = word_tokenize(line.lower())
            if not words:
                continue
                
            last_word = words[-1]
            if last_word in self.cmudict:
                # Get the last syllable's pronunciation
                pronunciation = self.cmudict[last_word][0]
                rhyme_scheme.append(pronunciation[-2:])  # Last two phonemes
        
        # Analyze rhyme patterns
        rhyme_counter = Counter(rhyme_scheme)
        most_common_rhymes = rhyme_counter.most_common(3)
        
        return {
            'rhyme_density': len(rhyme_scheme) / len(lines) if lines else 0,
            'common_rhymes': most_common_rhymes,
            'rhyme_variety': len(set(rhyme_scheme)) / len(rhyme_scheme) if rhyme_scheme else 0
        }
    
    def analyze_flow(self, lyrics: str) -> dict:
        """Analyze the flow characteristics of the lyrics.
        
        Args:
            lyrics (str): The lyrics to analyze
            
        Returns:
            dict: Flow characteristics
        """
        flow_stats = {}
        
        # Analyze word length patterns
        words = word_tokenize(lyrics.lower())
        word_lengths = [len(word) for word in words]
        flow_stats['avg_word_length'] = np.mean(word_lengths)
        flow_stats['word_length_variety'] = np.std(word_lengths)
        
        # Analyze flow patterns
        for pattern_name, pattern in self.flow_patterns.items():
            matches = re.findall(pattern, lyrics.lower())
            flow_stats[f'{pattern_name}_density'] = len(matches) / len(words) if words else 0
        
        return flow_stats
    
    def analyze_content(self, lyrics: str) -> dict:
        """Analyze the content characteristics of the lyrics.
        
        Args:
            lyrics (str): The lyrics to analyze
            
        Returns:
            dict: Content characteristics
        """
        blob = TextBlob(lyrics)
        
        # Sentiment analysis
        sentiment = blob.sentiment
        
        # Topic analysis
        words = word_tokenize(lyrics.lower())
        word_freq = Counter(words)
        
        # Common themes
        themes = {
            'violence': ['gun', 'shoot', 'kill', 'blood', 'death'],
            'wealth': ['money', 'cash', 'rich', 'wealth', 'dollar'],
            'success': ['win', 'success', 'victory', 'champion'],
            'struggle': ['pain', 'struggle', 'hardship', 'fight'],
            'social': ['society', 'world', 'people', 'life']
        }
        
        theme_scores = {}
        for theme, keywords in themes.items():
            score = sum(word_freq[word] for word in keywords)
            theme_scores[theme] = score / len(words) if words else 0
        
        return {
            'sentiment': {
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity
            },
            'theme_scores': theme_scores,
            'vocabulary_richness': len(set(words)) / len(words) if words else 0
        }

def extract_style_reference(artist_name: str, lyrics: str) -> str:
    """Extract style reference from lyrics and artist information.
    
    This function performs advanced analysis of the lyrics and artist name to create
    a comprehensive style reference that captures the artist's unique characteristics
    and the specific features of the given lyrics.
    
    Args:
        artist_name (str): Name of the artist (e.g., "Eminem", "Kendrick Lamar")
        lyrics (str): The lyrics to analyze for style characteristics
        
    Returns:
        str: A style reference string containing the artist's style characteristics
            and additional features found in the lyrics
            
    Example:
        >>> extract_style_reference("eminem", "Look, I was gonna go easy on you...")
        "Eminem's style: Fast-paced, complex wordplay, aggressive tone, and dark humor, with extensive vocabulary"
    """
    # Base style references
    style_references = {
        "eminem": "Fast-paced, complex wordplay, aggressive tone, and dark humor",
        "kendrick": "Conscious lyrics, complex storytelling, and social commentary",
        "jay-z": "Business-oriented, confident, and sophisticated wordplay",
        "nas": "Street poetry, vivid imagery, and introspective storytelling",
        "tupac": "Emotional depth, social consciousness, and raw authenticity",
        "kanye": "Innovative, self-reflective, and boundary-pushing",
        "drake": "Melodic flow, emotional vulnerability, and pop sensibilities",
        "j-cole": "Thoughtful, introspective, and socially conscious",
        "travis": "Atmospheric, psychedelic, and modern trap influence",
        "lil-wayne": "Creative wordplay, punchlines, and unique metaphors"
    }
    
    # Get the base style reference
    base_style = style_references.get(artist_name.lower(), "Unique flow and style")
    
    # Initialize style analyzer
    analyzer = StyleAnalyzer()
    
    # Perform advanced analysis
    rhyme_analysis = analyzer.analyze_rhyme_scheme(lyrics)
    flow_analysis = analyzer.analyze_flow(lyrics)
    content_analysis = analyzer.analyze_content(lyrics)
    
    # Extract characteristics based on analysis
    characteristics = []
    
    # Flow characteristics
    if flow_analysis['fast_density'] > 0.3:
        characteristics.append("fast-paced delivery")
    if flow_analysis['complex_density'] > 0.2:
        characteristics.append("complex wordplay")
    if flow_analysis['word_length_variety'] > 2.0:
        characteristics.append("varied flow patterns")
    
    # Rhyme characteristics
    if rhyme_analysis['rhyme_density'] > 0.6:
        characteristics.append("dense rhyming")
    if rhyme_analysis['rhyme_variety'] > 0.7:
        characteristics.append("diverse rhyme schemes")
    
    # Content characteristics
    if content_analysis['vocabulary_richness'] > 0.6:
        characteristics.append("extensive vocabulary")
    if content_analysis['sentiment']['subjectivity'] > 0.6:
        characteristics.append("personal expression")
    if content_analysis['theme_scores']['social'] > 0.1:
        characteristics.append("social commentary")
    if content_analysis['theme_scores']['struggle'] > 0.1:
        characteristics.append("emotional depth")
    
    # Combine characteristics
    if characteristics:
        style_reference = f"{base_style}, with {', '.join(characteristics)}"
    else:
        style_reference = base_style
    
    return style_reference

def prepare_style_dataset() -> None:
    """Prepare the style-based training dataset.
    
    This function processes the rap lyrics dataset to create a style-based training dataset.
    It loads the dataset, extracts style references for each song, and saves the processed
    data to a JSON file.
    
    The function performs the following steps:
    1. Loads the rap lyrics dataset from Hugging Face
    2. Processes each song to extract style characteristics
    3. Creates a JSON file with processed data including:
       - Artist name
       - Original lyrics
       - Style reference
       - Advanced style analysis
    4. Saves the dataset to 'data/rap_style_dataset.json'
    
    The output JSON file will have the following structure:
    [
        {
            "artist": "artist_name",
            "lyrics": "original_lyrics",
            "style_reference": "extracted_style_reference",
            "style_analysis": {
                "rhyme_analysis": {...},
                "flow_analysis": {...},
                "content_analysis": {...}
            }
        },
        ...
    ]
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If the output directory cannot be created
        json.JSONDecodeError: If there's an error saving the JSON file
    """
    # Load the rap lyrics dataset
    dataset = load_dataset("nateraw/rap-lyrics-v2", split='train')
    
    # Initialize style analyzer
    analyzer = StyleAnalyzer()
    
    # Process the data
    processed_data = []
    
    for item in tqdm(dataset, desc="Processing lyrics"):
        # Extract artist name and lyrics
        artist = item.get('artist', 'unknown')
        lyrics = item.get('text', '')
        
        # Skip if no lyrics
        if not lyrics:
            continue
        
        # Extract style reference
        style_reference = extract_style_reference(artist, lyrics)
        
        # Perform advanced analysis
        style_analysis = {
            'rhyme_analysis': analyzer.analyze_rhyme_scheme(lyrics),
            'flow_analysis': analyzer.analyze_flow(lyrics),
            'content_analysis': analyzer.analyze_content(lyrics)
        }
        
        # Create processed item
        processed_item = {
            'artist': artist,
            'lyrics': lyrics,
            'style_reference': style_reference,
            'style_analysis': style_analysis
        }
        
        processed_data.append(processed_item)
    
    # Save the processed data
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'rap_style_dataset.json')
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(processed_data)} songs")
    print(f"Saved dataset to {output_path}")

if __name__ == "__main__":
    prepare_style_dataset() 