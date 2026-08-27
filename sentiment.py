import re

# Conditional import of NLTK for robust fallback execution
try:
    import nltk
    # Download vader lexicon quietly if needed
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True)
        except Exception:
            pass
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

class SentimentAnalyzer:
    def __init__(self):
        self.use_fallback = not HAS_NLTK
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            self.use_fallback = True
            
        # Dictionary of emotive words for fallback analysis
        self.emotional_words = {
            'love': 0.8, 'hate': 0.8, 'angry': 0.7, 'furious': 0.9, 'sad': 0.6,
            'happy': 0.7, 'joy': 0.8, 'excited': 0.8, 'wonderful': 0.7, 'terrible': 0.8,
            'horrible': 0.8, 'great': 0.5, 'bad': 0.5, 'proposal': 0.7, 'proposed': 0.8,
            'fight': 0.7, 'argued': 0.6, 'cried': 0.7, 'amazing': 0.8, 'depressed': 0.8,
            'elated': 0.9, 'scared': 0.7, 'fear': 0.7, 'terrified': 0.9, 'delighted': 0.8,
            'broken': 0.6, 'hurt': 0.6, 'killed': 0.9, 'died': 0.9, 'win': 0.6, 'won': 0.7,
            'fail': 0.6, 'failed': 0.7, 'success': 0.6, 'marriage': 0.7, 'divorce': 0.8
        }

    def get_salience(self, text: str) -> float:
        """
        Extract emotional intensity/salience score S in [0, 1] from text.
        We take the absolute value of the compound score.
        """
        if not self.use_fallback:
            try:
                scores = self.sia.polarity_scores(text)
                return abs(scores['compound'])
            except Exception:
                # Fallback if VADER fails at runtime
                return self._fallback_salience(text)
        else:
            return self._fallback_salience(text)

    def _fallback_salience(self, text: str) -> float:
        """
        Rule-based fallback for emotional intensity.
        Uses simple keyword matching and punctuation cues.
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Base salience from emotional words
        salience = 0.0
        matches = 0
        for word in words:
            if word in self.emotional_words:
                salience += self.emotional_words[word]
                matches += 1
                
        if matches > 0:
            salience = min(1.0, salience / matches)
            
        # Punctuation multiplier (exclamation marks increase emotional intensity)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            salience = min(1.0, salience + 0.1 * min(exclamation_count, 3))
            
        # All-caps words (excluding small ones) indicate shouting/intensity
        caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]
        if caps_words:
            salience = min(1.0, salience + 0.15)
            
        # Default salience for neutral texts
        if salience == 0.0:
            # Slightly higher if it is long, but capped very low
            salience = min(0.15, len(text) / 2000.0)
            
        return round(salience, 3)

# Quick self-test if run directly
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print("Happy test:", analyzer.get_salience("I got proposed to today! I'm so happy!"))
    print("Angry test:", analyzer.get_salience("I had a massive fight with my brother today"))
    print("Neutral test:", analyzer.get_salience("The weather is 25 degrees and the wind is coming from the north."))
