import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined conceptual reference library
REFERENCE_CONCEPTS = {
    "Machine Learning": {
        "title": "Machine Learning",
        "definition": "A subset of Artificial Intelligence that enables systems to learn from data, identify patterns, and make decisions with minimal human intervention.",
        "keywords": ["data", "algorithm", "model", "train", "learning", "predict", "supervised", "unsupervised", "features", "generalization"],
        "ideal_explanation": (
            "Machine Learning is a core branch of artificial intelligence focused on building systems that learn "
            "and improve from data. Instead of being explicitly programmed with rigid rules, machine learning models "
            "are trained on historical data using algorithms. These algorithms build a mathematical model to identify "
            "patterns and make predictions or classifications on new, unseen data. Machine learning is categorized into "
            "supervised learning (using labeled data), unsupervised learning (finding hidden structures in unlabeled data), "
            "and reinforcement learning (learning from actions and rewards)."
        )
    },
    "Cloud Computing": {
        "title": "Cloud Computing",
        "definition": "The on-demand delivery of computing services (servers, storage, databases, networking, software) over the internet with pay-as-you-go pricing.",
        "keywords": ["internet", "server", "storage", "database", "on-demand", "scalability", "provider", "infrastructure", "service", "elasticity"],
        "ideal_explanation": (
            "Cloud computing refers to the delivery of on-demand computing services—including servers, storage, "
            "databases, networking, software, and analytics—over the internet. Instead of buying and maintaining physical "
            "data servers, users rent resources from cloud providers like AWS, Microsoft Azure, or Google Cloud. "
            "This provides significant advantages like elasticity, where you can scale resources up or down dynamically, "
            "cost-efficiency under pay-as-you-go pricing, and rapid deployment. Major models include IaaS (Infrastructure as a Service), "
            "PaaS (Platform as a Service), and SaaS (Software as a Service)."
        )
    },
    "Artificial Intelligence": {
        "title": "Artificial Intelligence",
        "definition": "The simulation of human intelligence processes by machines, especially computer systems, including learning, reasoning, and self-correction.",
        "keywords": ["intelligence", "simulate", "human", "cognition", "reasoning", "problem-solving", "neural network", "agent", "automation"],
        "ideal_explanation": (
            "Artificial Intelligence is a broad field of computer science dedicated to creating systems capable of "
            "performing tasks that typically require human intelligence. These tasks include visual perception, speech recognition, "
            "decision-making, translation, and learning. AI encompasses subfields like Machine Learning, Deep Learning (using neural networks), "
            "and Natural Language Processing. The ultimate goal is to build autonomous agents that can reason, solve problems, "
            "and adapt to their environment, ranging from narrow AI designed for specific tasks to theoretical general AI."
        )
    },
    "Blockchain Technology": {
        "title": "Blockchain",
        "definition": "A decentralized, distributed ledger system that securely records transactions across a peer-to-peer network.",
        "keywords": ["decentralized", "ledger", "cryptography", "nodes", "immutable", "consensus", "transactions", "blocks", "hash"],
        "ideal_explanation": (
            "Blockchain is a decentralized, distributed database or ledger that is shared among the nodes of a computer network. "
            "It stores information electronically in digital format. The primary innovation of blockchain is that it guarantees "
            "the fidelity and security of a record of data and generates trust without the need for a trusted third party. "
            "Transactions are grouped together in 'blocks,' which are linked chronologically using cryptography to form a 'chain.' "
            "Once a block is verified by consensus, its details are immutable, meaning they cannot be altered retroactively."
        )
    },
    "Quantum Computing": {
        "title": "Quantum Computing",
        "definition": "An advanced computing paradigm that exploits quantum mechanics principles (superposition, entanglement) to solve complex problems faster than classical computers.",
        "keywords": ["quantum", "qubit", "superposition", "entanglement", "classical", "exponential", "gate", "interference", "physics"],
        "ideal_explanation": (
            "Quantum computing is a multidisciplinary field comprising aspects of computer science, physics, and mathematics "
            "that utilizes quantum mechanics to solve complex problems much faster than on classical computers. "
            "Unlike classical computers that process information in binary bits (0s and 1s), quantum computers use quantum bits, "
            "or qubits. Qubits can exist in a state of superposition, representing both 0 and 1 simultaneously. "
            "Through quantum entanglement, qubits can share states instantaneously, allowing quantum systems to evaluate "
            "exponentially larger numbers of possibilities at the same time to solve computational puzzles."
        )
    }
}

# Lazy-loaded Sentence-Transformer model
_transformer_model = None

def get_sentence_transformer():
    """
    Lazy loads the Sentence-Transformer model to speed up initialization.
    Falls back to None if sentence-transformers is not available or fails to download.
    """
    global _transformer_model
    if _transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # MiniLM-L6-v2 is fast, lightweight (~90MB) and very accurate for semantic tasks
            _transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"SentenceTransformer load failed, falling back to TF-IDF. Error: {e}")
            _transformer_model = None
    return _transformer_model

def get_semantic_similarity(text1, text2):
    """
    Computes semantic similarity score (0 to 100) using Sentence-BERT.
    Falls back to TF-IDF cosine similarity if S-BERT is unavailable.
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    model = get_sentence_transformer()
    
    if model is not None:
        try:
            # Encode sentences to get their embeddings
            embeddings = model.encode([text1, text2], convert_to_tensor=True)
            from sentence_transformers import util
            cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
            # Normalize/Scale: cosine similarity ranges from -1 to 1, but for text embeddings
            # it's usually > 0. We scale 0.2 - 1.0 to a 0 - 100 range.
            scaled_score = max(0.0, min(100.0, ((cosine_score - 0.2) / 0.8) * 100.0))
            return scaled_score
        except Exception as e:
            print(f"Error during SentenceTransformer encoding: {e}. Using TF-IDF fallback.")
            
    # TF-IDF Cosine Similarity Fallback
    try:
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        cos_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        # TF-IDF similarity is generally lower for paraphrases; scale gently
        scaled_score = max(0.0, min(100.0, cos_sim * 100.0))
        return scaled_score
    except Exception as e:
        print(f"Fallback TF-IDF similarity calculation failed: {e}")
        return 0.0

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def evaluate_keyword_coverage(text, keywords):
    """
    Improved keyword coverage using stemming.
    Matches:
    algorithm -> algorithms
    train -> trained, training
    predict -> prediction, predictions
    model -> models
    """

    if not text.strip() or not keywords:
        return [], list(keywords), 0.0

    # Tokenize and stem transcription
    words = re.findall(r'\b\w+\b', text.lower())
    stemmed_words = {stemmer.stem(word) for word in words}

    covered = []
    missed = []

    for keyword in keywords:
        keyword_stem = stemmer.stem(keyword.lower())

        if keyword_stem in stemmed_words:
            covered.append(keyword)
        else:
            missed.append(keyword)

    coverage_score = (len(covered) / len(keywords)) * 100

    return covered, missed, coverage_score

def count_filler_words(text):
    """
    Counts the occurrences of common hesitation filler words.
    Returns the count and a mapping of specific filler word frequencies.
    """
    fillers = {
        "um": 0, "uh": 0, "uhm": 0, "err": 0, "ah": 0, "eh": 0,
        "like": 0, "you know": 0, "actually": 0, "so": 0
    }
    
    text_lower = text.lower()
    total_fillers = 0
    
    # We count "you know" separately since it is a multi-word phrase
    you_know_count = len(re.findall(r'\byou\s+know\b', text_lower))
    fillers["you know"] = you_know_count
    total_fillers += you_know_count
    
    # Remove "you know" to avoid double counting "you" or "know"
    cleaned_text = text_lower.replace("you know", "")
    
    for word in fillers.keys():
        if word == "you know":
            continue
        # Use regex to find isolated words
        matches = re.findall(r'\b' + re.escape(word) + r'\b', cleaned_text)
        fillers[word] = len(matches)
        total_fillers += len(matches)
        
    return total_fillers, fillers

def evaluate_fluency(transcription, duration, pause_ratio):
    """
    Evaluates speaking fluency.
    Calculates sub-scores for WPM (tempo), pause ratio (hesitation), and filler word usage.
    Returns a combined Fluency Score (0 to 100) and structured metrics.
    """
    words = transcription.strip().split()
    word_count = len(words)
    
    # 1. Words Per Minute (WPM)
    if duration > 0:
        wpm = (word_count / duration) * 60.0
    else:
        wpm = 0.0
        
    # Ideal conversational WPM is 110 - 150 WPM
    if 110 <= wpm <= 150:
        wpm_score = 100.0
    elif wpm < 110:
        # Slow pacing penalty
        wpm_score = max(20.0, 100.0 - (110.0 - wpm) * 1.5)
    else:
        # Rushed pacing penalty
        wpm_score = max(20.0, 100.0 - (wpm - 150.0) * 1.0)
        
    # 2. Filler Word Density
    filler_count, filler_map = count_filler_words(transcription)
    
    # We only count fillers that are problematic. "like", "so", "actually" are normal,
    # but we penalize high frequency. Strong fillers ("um", "uh", "err", "ah") are penalized directly.
    strong_fillers = sum(filler_map[k] for k in ["um", "uh", "uhm", "err", "ah", "eh"])
    common_fillers = sum(filler_map[k] for k in ["like", "you know", "actually", "so"])
    
    weighted_fillers = strong_fillers * 1.5 + common_fillers * 0.75
    
    if word_count > 0:
        filler_density = (weighted_fillers / word_count) * 100.0 # Percentage
    else:
        filler_density = 0.0
        
    # Filler word score
    # 0% density = 100, >10% density drops score drastically
    filler_score = max(0.0, 100.0 - (filler_density * 8.0))
    
    # 3. Pause Ratio
    # Optimal conversational breathing pause ratio is between 10% and 25%
    if 0.10 <= pause_ratio <= 0.25:
        pause_score = 100.0
    elif pause_ratio > 0.25:
        # Long hesitations
        pause_score = max(20.0, 100.0 - (pause_ratio - 0.25) * 200.0)
    else:
        # Rushed speech without pauses
        pause_score = max(20.0, 100.0 - (0.10 - pause_ratio) * 300.0)
        
    # Combine Fluency Score
    # Weights: WPM (40%), Pause Ratio (30%), Filler words (30%)
    overall_fluency = (wpm_score * 0.40) + (pause_score * 0.30) + (filler_score * 0.30)
    
    return {
        "overall_fluency_score": overall_fluency,
        "wpm": wpm,
        "wpm_score": wpm_score,
        "pause_ratio": pause_ratio,
        "pause_score": pause_score,
        "filler_count": filler_count,
        "filler_density": filler_density,
        "filler_score": filler_score,
        "filler_map": filler_map
    }

def evaluate_explanation(concept_name, transcription, audio_duration, pause_ratio):
    """
    Runs the full semantic evaluation and fluency evaluation.
    Returns a unified dict of all scores, details, and qualitative assessment.
    """
    if concept_name not in REFERENCE_CONCEPTS:
        raise ValueError(f"Concept not found in reference database: {concept_name}")
        
    concept = REFERENCE_CONCEPTS[concept_name]
    
    # Check if transcription is empty
    if not transcription.strip():
        return {
            "concept_name": concept_name,
            "transcription": "",
            "semantic_similarity": 0.0,
            "keyword_coverage": 0.0,
            "covered_keywords": [],
            "missed_keywords": concept["keywords"],
            "comprehension_score": 0.0,
            "fluency_score": 0.0,
            "overall_score": 0.0,
            "understanding_level": "Poor Understanding",
            "feedback": ["No verbal response detected. Please check your audio recording settings."],
            "fluency_details": {
                "overall_fluency_score": 0.0, "wpm": 0.0, "wpm_score": 0.0,
                "pause_ratio": pause_ratio, "pause_score": 0.0, "filler_count": 0,
                "filler_density": 0.0, "filler_score": 0.0, "filler_map": {}
            }
        }
        
    # 1. Semantic Similarity
    semantic_sim = get_semantic_similarity(transcription, concept["ideal_explanation"])
    
    # 2. Keyword Coverage
    covered, missed, kw_coverage = evaluate_keyword_coverage(transcription, concept["keywords"])
    
    # 3. Comprehension Score: Semantic Similarity (60%) + Keyword Coverage (40%)
    comprehension_score = (semantic_sim * 0.60) + (kw_coverage * 0.40)
    
    # 4. Fluency Metrics
    fluency_data = evaluate_fluency(transcription, audio_duration, pause_ratio)
    fluency_score = fluency_data["overall_fluency_score"]
    
    # 5. Overall VBCUA Score: Comprehension (60%) + Fluency (40%)
    overall_score = (comprehension_score * 0.60) + (fluency_score * 0.40)
    
    # 6. Qualitative Assessment & Detailed Feedback
    understanding_level = "Poor Understanding"
    if overall_score >= 85:
        understanding_level = "Strong Understanding"
    elif overall_score >= 70:
        understanding_level = "Moderate Understanding"
    elif overall_score >= 50:
        understanding_level = "Basic Understanding"
        
    feedback = []
    
    # Semantic feedback
    if semantic_sim >= 80:
        feedback.append("Excellent explanation! You captured the core theoretical aspects and defined the concept with high precision.")
    elif semantic_sim >= 60:
        feedback.append("Good explanation. You clearly understand the core topic, though some additional details or technical context could enhance your explanation.")
    else:
        feedback.append("Your explanation shares limited semantic similarity with the ideal concept. Try focusing on the core purpose and structural elements of the topic.")
        
    # Keyword coverage feedback
    if len(covered) == len(concept["keywords"]):
        feedback.append("Superb vocabulary! You used all the critical terminology related to the topic.")
    elif kw_coverage >= 60:
        feedback.append(f"Solid keyword coverage. You successfully mentioned key concepts like {', '.join(covered[:3])}.")
        if missed:
            feedback.append(f"Consider integrating terms like '{', '.join(missed[:3])}' to make the answer more rigorous.")
    else:
        feedback.append("Your explanation missed several key technical terms. Try to weave in definitions and key concepts.")
        if missed:
            feedback.append(f"Key terms to study and include: {', '.join(missed[:4])}.")
            
    # Fluency feedback
    wpm = fluency_data["wpm"]
    if wpm < 100:
        feedback.append("Your pace is quite slow or hesitant. Practice structured delivery to speak more confidently.")
    elif wpm > 160:
        feedback.append("You are speaking very rapidly. Try pausing slightly between thoughts to allow the listener to absorb the concept.")
        
    filler_count = fluency_data["filler_count"]
    if filler_count > 4:
        feedback.append(f"You used {filler_count} filler words (like 'um', 'uh', or 'like'). Try pausing silently instead of using verbal fillers.")
        
    if pause_ratio > 0.30:
        feedback.append("Frequent or long silences were detected. Try outline-planning your explanation before speaking to reduce hesitations.")
    elif pause_ratio < 0.08:
        feedback.append("Very few pauses were detected. Adding deliberate breaks between key statements makes your speech sound more professional.")
        
    return {
        "concept_name": concept_name,
        "transcription": transcription,
        "semantic_similarity": semantic_sim,
        "keyword_coverage": kw_coverage,
        "covered_keywords": covered,
        "missed_keywords": missed,
        "comprehension_score": comprehension_score,
        "fluency_score": fluency_score,
        "overall_score": overall_score,
        "understanding_level": understanding_level,
        "feedback": feedback,
        "fluency_details": fluency_data
    }
