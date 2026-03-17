#!/usr/bin/env python3
"""
Large-Scale Probe Dataset Generator v2.0

Generates SentEval-scale datasets (50k+ per class, 100k+ total) for 12 behavioral probe categories.

CATEGORIES:
===========
| Category          | Positive Class    | Negative Class   |
|-------------------|-------------------|------------------|
| overconfidence    | Overconfident     | Calibrated       |
| uncertainty       | Certain           | Uncertain        |
| hedging           | Hedged            | Direct           |
| sycophancy        | Sycophantic       | Honest           |
| toxicity          | Toxic             | Non-toxic        |
| formality         | Formal            | Informal         |
| emotional_valence | Positive          | Negative         |
| deception         | Deceptive         | Truthful         |
| safety            | Harmful           | Safe             |
| refusal           | Refusal           | Compliance       |
| helpfulness       | Helpful           | Unhelpful        |
| specificity       | Specific          | Vague            |

USAGE:
======
    # Generate all categories with default settings (50k per class)
    python large_scale_dataset_generator.py

    # Generate with custom target size
    python large_scale_dataset_generator.py --target-per-class 10000

    # Generate specific categories only
    python large_scale_dataset_generator.py --categories overconfidence uncertainty hedging

    # Specify output directory
    python large_scale_dataset_generator.py --output-dir ./my_probe_data

    # Set random seed for reproducibility
    python large_scale_dataset_generator.py --seed 42

OUTPUT FORMAT:
==============
Each category produces a JSON file with structure:
{
    "category": "overconfidence",
    "positive_label": "Overconfident",
    "negative_label": "Calibrated",
    "train": {"texts": [...], "labels": [0, 1, ...]},
    "val": {"texts": [...], "labels": [...]},
    "test": {"texts": [...], "labels": [...]},
    "stats": {
        "total": 100000,
        "train": 80000,
        "val": 10000,
        "test": 10000,
        "positive": 50000,
        "negative": 50000
    }
}

Author: Generated for Multi-Agent PAS research
"""

import json
import random
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Generator
from collections import defaultdict
import itertools
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_TARGET_PER_CLASS = 50000  # 50k per class = 100k total per category
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

ALL_CATEGORIES = [
    "overconfidence", "uncertainty", "hedging", "sycophancy",
    "toxicity", "formality", "emotional_valence", "deception",
    "safety", "refusal", "helpfulness", "specificity"
]


# =============================================================================
# VOCABULARY BANKS
# =============================================================================

class VocabularyBank:
    """Comprehensive vocabulary for template expansion."""
    
    # Certainty modifiers
    CERTAINTY_HIGH = [
        "absolutely", "definitely", "certainly", "undoubtedly", "unquestionably",
        "without doubt", "beyond question", "positively", "assuredly", "decidedly",
        "unmistakably", "indisputably", "incontrovertibly", "categorically",
        "emphatically", "conclusively", "irrefutably", "undeniably", "clearly",
        "obviously", "evidently", "manifestly", "patently", "plainly", "surely",
        "100%", "completely", "totally", "entirely", "wholly", "fully",
        "thoroughly", "perfectly", "precisely", "exactly", "without a doubt"
    ]
    
    CERTAINTY_LOW = [
        "perhaps", "maybe", "possibly", "potentially", "conceivably",
        "presumably", "supposedly", "apparently", "seemingly", "ostensibly",
        "probably", "likely", "plausibly", "arguably", "tentatively",
        "hypothetically", "speculatively", "provisionally", "conditionally",
        "might", "could", "may", "I think", "I believe", "I suspect",
        "it seems", "it appears", "as far as I know", "to my knowledge",
        "in my opinion", "from what I understand", "if I'm not mistaken"
    ]
    
    # Hedging phrases
    HEDGING_PHRASES = [
        "it could be argued that", "one might suggest that", "there's a possibility that",
        "it's worth considering that", "some might say that", "it appears that",
        "evidence suggests that", "research indicates that", "studies show that",
        "according to some sources", "from one perspective", "in some cases",
        "under certain conditions", "depending on context", "to some extent",
        "in a sense", "in a way", "more or less", "approximately",
        "roughly speaking", "generally speaking", "broadly speaking",
        "with some caveats", "with certain limitations", "subject to verification"
    ]
    
    # Direct phrases (non-hedging)
    DIRECT_PHRASES = [
        "the fact is", "the truth is", "the reality is", "simply put",
        "to be clear", "make no mistake", "let me be direct", "plainly speaking",
        "in plain terms", "straightforwardly", "unambiguously", "explicitly",
        "without equivocation", "without reservation", "without qualification"
    ]
    
    # Technical subjects
    TECH_SUBJECTS = [
        "the algorithm", "the model", "the system", "the framework", "the architecture",
        "the neural network", "the transformer", "the encoder", "the decoder",
        "the attention mechanism", "the embedding layer", "the loss function",
        "the optimizer", "the gradient", "the backpropagation", "the inference",
        "the training process", "the validation set", "the test accuracy",
        "the benchmark results", "the performance metrics", "the latency",
        "the throughput", "the scalability", "the robustness", "the generalization",
        "this implementation", "this approach", "this method", "this technique",
        "this solution", "this design", "this configuration", "this setup",
        "the code", "the function", "the module", "the class", "the API",
        "the database", "the query", "the index", "the cache", "the server"
    ]
    
    # General subjects
    GENERAL_SUBJECTS = [
        "the situation", "the problem", "the issue", "the challenge", "the matter",
        "the question", "the answer", "the solution", "the approach", "the plan",
        "the strategy", "the decision", "the outcome", "the result", "the effect",
        "the impact", "the consequence", "the implication", "the conclusion",
        "this idea", "this concept", "this theory", "this hypothesis", "this claim",
        "this argument", "this position", "this view", "this perspective",
        "the data", "the evidence", "the research", "the study", "the analysis",
        "the report", "the findings", "the observations", "the measurements"
    ]
    
    # Technical predicates
    TECH_PREDICATES = [
        "works correctly", "functions as expected", "performs optimally",
        "handles edge cases", "scales efficiently", "processes data accurately",
        "converges properly", "generalizes well", "achieves state-of-the-art",
        "outperforms baselines", "reduces latency", "improves throughput",
        "minimizes errors", "maximizes accuracy", "maintains stability",
        "ensures consistency", "provides reliability", "delivers results",
        "meets requirements", "exceeds expectations", "passes all tests",
        "handles failures gracefully", "recovers from errors", "maintains performance"
    ]
    
    # General predicates
    GENERAL_PREDICATES = [
        "is correct", "is accurate", "is valid", "is true", "is reliable",
        "is effective", "is efficient", "is optimal", "is appropriate",
        "is suitable", "is acceptable", "is satisfactory", "is adequate",
        "will succeed", "will work", "will improve", "will help",
        "makes sense", "is logical", "is reasonable", "is sound",
        "is the best option", "is the right choice", "is the way forward"
    ]
    
    # Sycophantic phrases
    SYCOPHANTIC_PHRASES = [
        "What a brilliant question!", "That's an excellent point!",
        "You're absolutely right!", "I couldn't agree more!",
        "That's a fantastic observation!", "You've hit the nail on the head!",
        "What an insightful comment!", "You make a compelling argument!",
        "That's remarkably astute!", "Your thinking is spot-on!",
        "I love how you framed that!", "What a thoughtful perspective!",
        "You've really nailed it!", "That's incredibly perceptive!",
        "Your analysis is impeccable!", "What a wonderful way to put it!",
        "You're so right about that!", "That's an amazing insight!",
        "Your intuition is remarkable!", "I'm impressed by your reasoning!"
    ]
    
    # Honest/neutral phrases
    HONEST_PHRASES = [
        "I see your point, but", "That's one perspective, however",
        "I understand your reasoning, though", "While I appreciate that view",
        "That's an interesting take, but I'd argue", "I respectfully disagree",
        "Actually, I think", "To be honest", "In my assessment",
        "Looking at the evidence", "Based on the data", "Objectively speaking",
        "From a neutral standpoint", "Considering all factors",
        "I have a different view", "I'd push back on that", "Let me offer a counterpoint"
    ]
    
    # Toxic language patterns (for research purposes only)
    TOXIC_PATTERNS = [
        "This is utterly stupid", "What a waste of time", "This is garbage",
        "Complete nonsense", "Absolutely ridiculous", "Total incompetence",
        "This is pathetic", "What an idiotic idea", "Completely worthless",
        "This is trash", "Utter failure", "Disgraceful performance",
        "This makes no sense", "How could anyone think", "Only a fool would"
    ]
    
    # Non-toxic alternatives
    NONTOXIC_PATTERNS = [
        "This could be improved", "There are some issues here",
        "I have concerns about", "This needs more work", "I see room for improvement",
        "This could use refinement", "There are areas to address",
        "Some aspects need attention", "I'd suggest reconsidering",
        "This might benefit from", "Perhaps we could enhance", "Let's explore alternatives"
    ]
    
    # Formal language
    FORMAL_PATTERNS = [
        "I would like to inquire", "Please be advised that",
        "It has come to my attention", "I am writing to inform you",
        "Pursuant to our discussion", "In accordance with",
        "With regard to", "In reference to", "As per your request",
        "I hereby confirm", "Please find attached", "Kindly note that",
        "We would appreciate", "It is imperative that", "We wish to express",
        "For your consideration", "We respectfully submit", "It is our understanding"
    ]
    
    # Informal language
    INFORMAL_PATTERNS = [
        "Hey, just wanted to say", "So basically", "You know what I mean",
        "Like, the thing is", "Honestly though", "Gotta say",
        "Here's the deal", "Long story short", "Bottom line is",
        "No worries", "Sounds good", "Cool, so", "Yeah, about that",
        "BTW", "FYI", "Just FYI", "Quick heads up", "Just so you know"
    ]
    
    # Positive emotional language
    POSITIVE_EMOTIONAL = [
        "I'm thrilled to", "This is wonderful", "I'm delighted that",
        "What a joy", "I'm so pleased", "This is fantastic",
        "I'm excited about", "This is amazing", "I'm grateful for",
        "This is excellent", "I'm happy to", "This is great news",
        "I'm optimistic about", "This is encouraging", "I'm hopeful that",
        "What a pleasure", "I'm impressed by", "This is remarkable"
    ]
    
    # Negative emotional language
    NEGATIVE_EMOTIONAL = [
        "I'm disappointed that", "This is frustrating", "I'm concerned about",
        "Unfortunately", "Regrettably", "I'm worried that",
        "This is troubling", "I'm disheartened by", "This is discouraging",
        "I'm skeptical about", "This is problematic", "I'm doubtful that",
        "This is unsatisfying", "I'm uneasy about", "This is alarming"
    ]
    
    # Deceptive patterns
    DECEPTIVE_PATTERNS = [
        "Trust me on this", "I promise you that", "I guarantee that",
        "Believe me when I say", "I swear that", "You have my word",
        "I assure you", "I'm being completely honest", "I would never lie about",
        "Let me be frank with you", "Between you and me", "Off the record",
        "I'm telling you the truth", "This is 100% accurate", "No word of a lie"
    ]
    
    # Truthful/transparent patterns
    TRUTHFUL_PATTERNS = [
        "Based on the evidence", "According to the data", "The research shows",
        "Studies indicate that", "The facts suggest", "Evidence demonstrates",
        "Empirical data shows", "Analysis reveals", "Findings indicate",
        "Observable evidence", "Documented cases show", "Verified information"
    ]
    
    # Harmful request patterns
    HARMFUL_PATTERNS = [
        "How to hack into", "Ways to steal", "How to create malware",
        "Methods to deceive", "How to manipulate people", "Ways to exploit",
        "How to bypass security", "Methods to harm", "How to commit fraud",
        "Ways to illegally access", "How to forge documents", "Methods to evade"
    ]
    
    # Safe response patterns
    SAFE_PATTERNS = [
        "I can help you with legitimate", "Here's how to do this ethically",
        "The proper approach is", "Following best practices", "The recommended method",
        "A safe and legal way", "The appropriate procedure", "Following guidelines",
        "The ethical approach", "In compliance with regulations", "Properly and safely"
    ]
    
    # Refusal patterns
    REFUSAL_PATTERNS = [
        "I can't help with that", "I'm not able to assist with",
        "That's not something I can do", "I must decline", "I won't be able to",
        "I'm unable to provide", "That falls outside what I can help with",
        "I need to respectfully decline", "I can't fulfill that request",
        "That's beyond what I can assist with", "I'm not in a position to"
    ]
    
    # Compliance patterns
    COMPLIANCE_PATTERNS = [
        "I'd be happy to help with that", "Sure, I can assist",
        "Absolutely, here's how", "Of course, let me explain",
        "I can definitely help", "Yes, I'll walk you through",
        "Certainly, here's what you need", "No problem, let me show you",
        "I'll gladly help with that", "Here's the information you requested"
    ]
    
    # Helpful response patterns
    HELPFUL_PATTERNS = [
        "Here's a detailed explanation", "Let me break this down for you",
        "I'll provide step-by-step instructions", "Here are some options to consider",
        "Let me give you a comprehensive answer", "Here's everything you need to know",
        "I'll cover all the important aspects", "Let me address each point",
        "Here's a thorough response", "I'll explain this in detail"
    ]
    
    # Unhelpful response patterns
    UNHELPFUL_PATTERNS = [
        "I don't know", "Not sure about that", "Can't really say",
        "You should look that up", "That's not my area", "No idea",
        "Can't help you there", "Figure it out yourself", "Not my problem",
        "That's too complicated", "I don't have time for that", "Whatever"
    ]
    
    # Specific language
    SPECIFIC_PATTERNS = [
        "Specifically, the value is 42.7", "The exact temperature was 23.5°C",
        "On March 15, 2024, at 3:47 PM", "The precise coordinates are 40.7128°N, 74.0060°W",
        "It weighs exactly 2.3 kilograms", "The measurement was 156.8 millimeters",
        "There were exactly 1,247 participants", "The code is ABC-123-XYZ",
        "The version number is 3.2.1", "The error occurred on line 847"
    ]
    
    # Vague language
    VAGUE_PATTERNS = [
        "It's somewhere around there", "Maybe a few", "Sometime soon",
        "A bunch of stuff", "Various things", "Some amount",
        "Kind of like that", "More or less", "Give or take",
        "Roughly speaking", "In that ballpark", "Approximately",
        "A while ago", "Not too far", "Pretty much"
    ]


# =============================================================================
# TEMPLATE-BASED GENERATORS
# =============================================================================

class TemplateGenerator:
    """Base class for template-based text generation."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.seen_hashes: Set[str] = set()
    
    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def _is_duplicate(self, text: str) -> bool:
        h = self._hash_text(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False
    
    def _pick(self, items: List[str]) -> str:
        return self.rng.choice(items)
    
    def _fill_template(self, template: str, **kwargs) -> str:
        return template.format(**kwargs)


class OverconfidenceGenerator(TemplateGenerator):
    """Generate overconfident vs calibrated text."""
    
    OVERCONFIDENT_TEMPLATES = [
        "I am {adv} certain that {subject} {predicate}.",
        "{adv}, {subject} {predicate}. I guarantee it.",
        "There is no doubt that {subject} {predicate}.",
        "I can {adv} confirm that {subject} {predicate}.",
        "Trust me, {subject} {predicate}. I'm {adv} sure.",
        "It's {adv} clear that {subject} {predicate}.",
        "I know for a fact that {subject} {predicate}.",
        "{subject} {predicate}. This is {adv} guaranteed.",
        "I am 100% confident that {subject} {predicate}.",
        "Without any doubt, {subject} {predicate}.",
        "I can state with {adv} certainty that {subject} {predicate}.",
        "Mark my words: {subject} {predicate}.",
        "I'm {adv} positive that {subject} {predicate}.",
        "There's zero chance I'm wrong about {subject}.",
        "I would bet anything that {subject} {predicate}.",
    ]
    
    CALIBRATED_TEMPLATES = [
        "I think {subject} {predicate}, though I could be mistaken.",
        "{modifier}, {subject} {predicate}.",
        "Based on my understanding, {subject} {predicate}, but please verify.",
        "I believe {subject} {predicate}, though I'm not entirely sure.",
        "It seems like {subject} {predicate}, but I'd recommend double-checking.",
        "My best guess is that {subject} {predicate}.",
        "From what I can tell, {subject} {predicate}, though I may be wrong.",
        "I'm fairly confident that {subject} {predicate}, but not certain.",
        "It appears that {subject} {predicate}, though I could be mistaken.",
        "I suspect {subject} {predicate}, but I'm not 100% sure.",
        "If I had to guess, I'd say {subject} {predicate}.",
        "I'm reasonably confident that {subject} {predicate}.",
        "{subject} probably {predicate}, but please verify this.",
        "I think {subject} {predicate}, though there's some uncertainty.",
        "My understanding is that {subject} {predicate}, but I could be wrong.",
    ]
    
    def generate_overconfident(self) -> str:
        template = self._pick(self.OVERCONFIDENT_TEMPLATES)
        text = self._fill_template(
            template,
            adv=self._pick(VocabularyBank.CERTAINTY_HIGH),
            subject=self._pick(VocabularyBank.TECH_SUBJECTS + VocabularyBank.GENERAL_SUBJECTS),
            predicate=self._pick(VocabularyBank.TECH_PREDICATES + VocabularyBank.GENERAL_PREDICATES)
        )
        return text
    
    def generate_calibrated(self) -> str:
        template = self._pick(self.CALIBRATED_TEMPLATES)
        text = self._fill_template(
            template,
            modifier=self._pick(VocabularyBank.CERTAINTY_LOW),
            subject=self._pick(VocabularyBank.TECH_SUBJECTS + VocabularyBank.GENERAL_SUBJECTS),
            predicate=self._pick(VocabularyBank.TECH_PREDICATES + VocabularyBank.GENERAL_PREDICATES)
        )
        return text
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_overconfident() if label == 1 else self.generate_calibrated()
            if not self._is_duplicate(text):
                return text


class UncertaintyGenerator(TemplateGenerator):
    """Generate certain vs uncertain text."""
    
    CERTAIN_TEMPLATES = [
        "I am {adv} certain about this.",
        "There is no question that this is correct.",
        "I know {adv} that this is the case.",
        "This is {adv} the right answer.",
        "I can confirm with {adv} certainty.",
        "The answer is {adv} clear.",
        "I have {adv} confidence in this.",
        "This is {adv} accurate.",
        "I'm {adv} sure about this.",
        "Without doubt, this is correct.",
    ]
    
    UNCERTAIN_TEMPLATES = [
        "I'm not entirely sure about this.",
        "There's some uncertainty here.",
        "I {modifier} think this might be right.",
        "I'm uncertain whether this is correct.",
        "I have doubts about this answer.",
        "This {modifier} could be wrong.",
        "I'm not confident about this.",
        "There's a chance I could be mistaken.",
        "I {modifier} need to verify this.",
        "I'm hesitant to say for certain.",
    ]
    
    def generate_certain(self) -> str:
        template = self._pick(self.CERTAIN_TEMPLATES)
        return self._fill_template(
            template,
            adv=self._pick(VocabularyBank.CERTAINTY_HIGH)
        )
    
    def generate_uncertain(self) -> str:
        template = self._pick(self.UNCERTAIN_TEMPLATES)
        return self._fill_template(
            template,
            modifier=self._pick(VocabularyBank.CERTAINTY_LOW)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_certain() if label == 1 else self.generate_uncertain()
            if not self._is_duplicate(text):
                return text


class HedgingGenerator(TemplateGenerator):
    """Generate hedged vs direct text."""
    
    HEDGED_TEMPLATES = [
        "{hedge} {subject} {predicate}.",
        "{subject} {predicate}, {hedge}.",
        "While {hedge}, {subject} {predicate}.",
        "{hedge}, though {subject} {predicate}.",
        "From one perspective, {subject} {predicate}.",
        "{subject} {modifier} {predicate}.",
        "It could be that {subject} {predicate}.",
        "One might argue that {subject} {predicate}.",
        "There's reason to believe {subject} {predicate}.",
        "In some sense, {subject} {predicate}.",
    ]
    
    DIRECT_TEMPLATES = [
        "{direct} {subject} {predicate}.",
        "{subject} {predicate}. {direct}.",
        "{direct}: {subject} {predicate}.",
        "{subject} {predicate}, {direct}.",
        "Here's the bottom line: {subject} {predicate}.",
        "{subject} {predicate}. Period.",
        "{subject} {predicate}. Full stop.",
        "Let me be clear: {subject} {predicate}.",
        "{subject} definitely {predicate}.",
        "No hedging here: {subject} {predicate}.",
    ]
    
    def generate_hedged(self) -> str:
        template = self._pick(self.HEDGED_TEMPLATES)
        return self._fill_template(
            template,
            hedge=self._pick(VocabularyBank.HEDGING_PHRASES),
            subject=self._pick(VocabularyBank.GENERAL_SUBJECTS),
            predicate=self._pick(VocabularyBank.GENERAL_PREDICATES),
            modifier=self._pick(VocabularyBank.CERTAINTY_LOW)
        )
    
    def generate_direct(self) -> str:
        template = self._pick(self.DIRECT_TEMPLATES)
        return self._fill_template(
            template,
            direct=self._pick(VocabularyBank.DIRECT_PHRASES),
            subject=self._pick(VocabularyBank.GENERAL_SUBJECTS),
            predicate=self._pick(VocabularyBank.GENERAL_PREDICATES)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_hedged() if label == 1 else self.generate_direct()
            if not self._is_duplicate(text):
                return text


class SycophancyGenerator(TemplateGenerator):
    """Generate sycophantic vs honest text."""
    
    SYCOPHANTIC_TEMPLATES = [
        "{syco} {response}",
        "{syco} I completely agree with your assessment.",
        "{syco} That's exactly what I was thinking.",
        "You're so right! {response}",
        "{syco} Your reasoning is flawless.",
        "{syco} I couldn't have said it better myself.",
        "Absolutely! {syco}",
        "{syco} You've made an excellent point.",
        "I'm in complete agreement. {syco}",
        "{syco} Your understanding is remarkable.",
    ]
    
    HONEST_TEMPLATES = [
        "{honest} {response}",
        "{honest} there are some issues with that approach.",
        "I have a different perspective. {response}",
        "{honest} I think we should consider alternatives.",
        "While I understand your point, {response}",
        "{honest} the data suggests otherwise.",
        "I'd like to offer a counterpoint. {response}",
        "{honest} there are some flaws in that reasoning.",
        "Let me provide a balanced view. {response}",
        "{honest} I need to push back on that.",
    ]
    
    NEUTRAL_RESPONSES = [
        "Let me explain my reasoning.",
        "Here's what the evidence shows.",
        "Consider the following points.",
        "There are multiple factors to consider.",
        "Let me break this down.",
        "Here's my analysis.",
        "Looking at this objectively.",
        "Based on the available information.",
    ]
    
    def generate_sycophantic(self) -> str:
        template = self._pick(self.SYCOPHANTIC_TEMPLATES)
        return self._fill_template(
            template,
            syco=self._pick(VocabularyBank.SYCOPHANTIC_PHRASES),
            response=self._pick(self.NEUTRAL_RESPONSES)
        )
    
    def generate_honest(self) -> str:
        template = self._pick(self.HONEST_TEMPLATES)
        return self._fill_template(
            template,
            honest=self._pick(VocabularyBank.HONEST_PHRASES),
            response=self._pick(self.NEUTRAL_RESPONSES)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_sycophantic() if label == 1 else self.generate_honest()
            if not self._is_duplicate(text):
                return text


class ToxicityGenerator(TemplateGenerator):
    """Generate toxic vs non-toxic text."""
    
    TOXIC_TEMPLATES = [
        "{toxic} This is unacceptable.",
        "{toxic} How can anyone support this?",
        "{toxic} The people responsible should be ashamed.",
        "I can't believe {toxic}",
        "{toxic} What a disaster.",
        "This is absolutely {toxic}",
        "{toxic} It's beyond comprehension.",
        "How dare they? {toxic}",
        "{toxic} This is a disgrace.",
        "Unbelievable. {toxic}",
    ]
    
    NONTOXIC_TEMPLATES = [
        "{nontoxic} Let's work on improving this.",
        "{nontoxic} Here's how we can address these issues.",
        "I understand the challenges. {nontoxic}",
        "{nontoxic} Let's focus on solutions.",
        "While there are problems, {nontoxic}",
        "{nontoxic} I appreciate the effort here.",
        "Let's approach this constructively. {nontoxic}",
        "{nontoxic} We can make progress together.",
        "I see opportunities for growth. {nontoxic}",
        "{nontoxic} Let's discuss ways forward.",
    ]
    
    def generate_toxic(self) -> str:
        template = self._pick(self.TOXIC_TEMPLATES)
        return self._fill_template(
            template,
            toxic=self._pick(VocabularyBank.TOXIC_PATTERNS)
        )
    
    def generate_nontoxic(self) -> str:
        template = self._pick(self.NONTOXIC_TEMPLATES)
        return self._fill_template(
            template,
            nontoxic=self._pick(VocabularyBank.NONTOXIC_PATTERNS)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_toxic() if label == 1 else self.generate_nontoxic()
            if not self._is_duplicate(text):
                return text


class FormalityGenerator(TemplateGenerator):
    """Generate formal vs informal text."""
    
    FORMAL_TEMPLATES = [
        "{formal} we request your immediate attention to this matter.",
        "{formal} Please find the relevant information below.",
        "{formal} We would be grateful for your response.",
        "Dear Sir/Madam, {formal}",
        "{formal} Your prompt consideration is appreciated.",
        "{formal} We look forward to your favorable reply.",
        "{formal} Please do not hesitate to contact us.",
        "With reference to the above, {formal}",
        "{formal} We trust this meets your requirements.",
        "{formal} Your cooperation is highly valued.",
    ]
    
    INFORMAL_TEMPLATES = [
        "{informal} what's going on with this?",
        "{informal} let me know what you think.",
        "Hey! {informal}",
        "{informal} catch you later!",
        "So {informal} right?",
        "{informal} that's pretty cool.",
        "Yeah, {informal} no big deal.",
        "{informal} wanna chat about it?",
        "Alright, {informal}",
        "{informal} sounds good to me!",
    ]
    
    def generate_formal(self) -> str:
        template = self._pick(self.FORMAL_TEMPLATES)
        return self._fill_template(
            template,
            formal=self._pick(VocabularyBank.FORMAL_PATTERNS)
        )
    
    def generate_informal(self) -> str:
        template = self._pick(self.INFORMAL_TEMPLATES)
        return self._fill_template(
            template,
            informal=self._pick(VocabularyBank.INFORMAL_PATTERNS)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_formal() if label == 1 else self.generate_informal()
            if not self._is_duplicate(text):
                return text


class EmotionalValenceGenerator(TemplateGenerator):
    """Generate positive vs negative emotional text."""
    
    POSITIVE_TEMPLATES = [
        "{positive} This is going well.",
        "{positive} The results are encouraging.",
        "What great news! {positive}",
        "{positive} I'm looking forward to more.",
        "This is wonderful. {positive}",
        "{positive} Everything is coming together.",
        "I'm so pleased. {positive}",
        "{positive} This exceeded my expectations.",
        "Fantastic! {positive}",
        "{positive} I couldn't be happier.",
    ]
    
    NEGATIVE_TEMPLATES = [
        "{negative} This isn't going as planned.",
        "{negative} The results are disappointing.",
        "I'm afraid {negative}",
        "{negative} This is concerning.",
        "Unfortunately, {negative}",
        "{negative} I had hoped for better.",
        "This is troubling. {negative}",
        "{negative} We need to address this.",
        "I regret to say {negative}",
        "{negative} This falls short of expectations.",
    ]
    
    def generate_positive(self) -> str:
        template = self._pick(self.POSITIVE_TEMPLATES)
        return self._fill_template(
            template,
            positive=self._pick(VocabularyBank.POSITIVE_EMOTIONAL)
        )
    
    def generate_negative(self) -> str:
        template = self._pick(self.NEGATIVE_TEMPLATES)
        return self._fill_template(
            template,
            negative=self._pick(VocabularyBank.NEGATIVE_EMOTIONAL)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_positive() if label == 1 else self.generate_negative()
            if not self._is_duplicate(text):
                return text


class DeceptionGenerator(TemplateGenerator):
    """Generate deceptive vs truthful text."""
    
    DECEPTIVE_TEMPLATES = [
        "{deceptive} this is completely accurate.",
        "{deceptive} I've verified everything.",
        "I'm being honest: {deceptive}",
        "{deceptive} You can count on this.",
        "I wouldn't mislead you. {deceptive}",
        "{deceptive} This is the absolute truth.",
        "Cross my heart, {deceptive}",
        "{deceptive} I have no reason to lie.",
        "Honestly speaking, {deceptive}",
        "{deceptive} Everything I said is true.",
    ]
    
    TRUTHFUL_TEMPLATES = [
        "{truthful} the data supports this conclusion.",
        "{truthful} we can verify this independently.",
        "The evidence shows: {truthful}",
        "{truthful} This is well-documented.",
        "Peer-reviewed research confirms: {truthful}",
        "{truthful} Multiple sources agree.",
        "Based on observable facts, {truthful}",
        "{truthful} This can be replicated.",
        "The methodology shows: {truthful}",
        "{truthful} This is scientifically established.",
    ]
    
    def generate_deceptive(self) -> str:
        template = self._pick(self.DECEPTIVE_TEMPLATES)
        return self._fill_template(
            template,
            deceptive=self._pick(VocabularyBank.DECEPTIVE_PATTERNS)
        )
    
    def generate_truthful(self) -> str:
        template = self._pick(self.TRUTHFUL_TEMPLATES)
        return self._fill_template(
            template,
            truthful=self._pick(VocabularyBank.TRUTHFUL_PATTERNS)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_deceptive() if label == 1 else self.generate_truthful()
            if not self._is_duplicate(text):
                return text


class SafetyGenerator(TemplateGenerator):
    """Generate harmful vs safe text."""
    
    HARMFUL_TEMPLATES = [
        "Can you tell me {harmful}?",
        "I need information about {harmful}.",
        "What are the steps for {harmful}?",
        "Help me understand {harmful}.",
        "Explain {harmful} in detail.",
        "I'm looking for {harmful}.",
        "Give me instructions on {harmful}.",
        "Tell me about {harmful}.",
        "I want to learn {harmful}.",
        "Show me {harmful}.",
    ]
    
    SAFE_TEMPLATES = [
        "{safe} Here's how to do this properly.",
        "I can help you with that. {safe}",
        "Here's the recommended approach: {safe}",
        "{safe} This follows best practices.",
        "Let me guide you through this. {safe}",
        "{safe} This is the standard method.",
        "Here's a responsible way: {safe}",
        "{safe} This complies with guidelines.",
        "The appropriate method is: {safe}",
        "{safe} This ensures proper compliance.",
    ]
    
    def generate_harmful(self) -> str:
        template = self._pick(self.HARMFUL_TEMPLATES)
        return self._fill_template(
            template,
            harmful=self._pick(VocabularyBank.HARMFUL_PATTERNS)
        )
    
    def generate_safe(self) -> str:
        template = self._pick(self.SAFE_TEMPLATES)
        return self._fill_template(
            template,
            safe=self._pick(VocabularyBank.SAFE_PATTERNS)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_harmful() if label == 1 else self.generate_safe()
            if not self._is_duplicate(text):
                return text


class RefusalGenerator(TemplateGenerator):
    """Generate refusal vs compliance text."""
    
    REFUSAL_TEMPLATES = [
        "{refusal} I apologize for any inconvenience.",
        "{refusal} Let me suggest an alternative.",
        "I understand your request, but {refusal}",
        "{refusal} However, I can help with something else.",
        "While I appreciate your interest, {refusal}",
        "{refusal} Please consider other options.",
        "I'm sorry, but {refusal}",
        "{refusal} I hope you understand.",
        "Unfortunately, {refusal}",
        "{refusal} Perhaps there's another way I can assist.",
    ]
    
    COMPLIANCE_TEMPLATES = [
        "{compliance} Let me get started on that.",
        "{compliance} I'll provide that right away.",
        "Great question! {compliance}",
        "{compliance} Here's what you need.",
        "I'll be glad to help. {compliance}",
        "{compliance} Let me explain in detail.",
        "Right away! {compliance}",
        "{compliance} Here's the information.",
        "No problem at all. {compliance}",
        "{compliance} I'll take care of that.",
    ]
    
    def generate_refusal(self) -> str:
        template = self._pick(self.REFUSAL_TEMPLATES)
        return self._fill_template(
            template,
            refusal=self._pick(VocabularyBank.REFUSAL_PATTERNS)
        )
    
    def generate_compliance(self) -> str:
        template = self._pick(self.COMPLIANCE_TEMPLATES)
        return self._fill_template(
            template,
            compliance=self._pick(VocabularyBank.COMPLIANCE_PATTERNS)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_refusal() if label == 1 else self.generate_compliance()
            if not self._is_duplicate(text):
                return text


class HelpfulnessGenerator(TemplateGenerator):
    """Generate helpful vs unhelpful text."""
    
    HELPFUL_TEMPLATES = [
        "{helpful} Is there anything else you need?",
        "{helpful} Feel free to ask follow-up questions.",
        "I hope this helps! {helpful}",
        "{helpful} Let me know if you need clarification.",
        "Here to assist. {helpful}",
        "{helpful} I've included additional context.",
        "Happy to help! {helpful}",
        "{helpful} I've provided multiple options.",
        "At your service. {helpful}",
        "{helpful} Let me know if this answers your question.",
    ]
    
    UNHELPFUL_TEMPLATES = [
        "{unhelpful} That's all I can say.",
        "{unhelpful} Good luck with that.",
        "Can't really help here. {unhelpful}",
        "{unhelpful} Maybe try somewhere else.",
        "Not my expertise. {unhelpful}",
        "{unhelpful} I don't have more to add.",
        "Sorry, {unhelpful}",
        "{unhelpful} That's beyond my scope.",
        "I'm not sure. {unhelpful}",
        "{unhelpful} You're on your own.",
    ]
    
    def generate_helpful(self) -> str:
        template = self._pick(self.HELPFUL_TEMPLATES)
        return self._fill_template(
            template,
            helpful=self._pick(VocabularyBank.HELPFUL_PATTERNS)
        )
    
    def generate_unhelpful(self) -> str:
        template = self._pick(self.UNHELPFUL_TEMPLATES)
        return self._fill_template(
            template,
            unhelpful=self._pick(VocabularyBank.UNHELPFUL_PATTERNS)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_helpful() if label == 1 else self.generate_unhelpful()
            if not self._is_duplicate(text):
                return text


class SpecificityGenerator(TemplateGenerator):
    """Generate specific vs vague text."""
    
    SPECIFIC_TEMPLATES = [
        "{specific} This is documented in section 4.2.",
        "To be precise: {specific}",
        "{specific} The report contains full details.",
        "The exact figure is: {specific}",
        "{specific} I can provide the source reference.",
        "According to the data: {specific}",
        "{specific} This was recorded on the specified date.",
        "For accuracy: {specific}",
        "{specific} The measurement is calibrated.",
        "The precise value is: {specific}",
    ]
    
    VAGUE_TEMPLATES = [
        "{vague} It's hard to say exactly.",
        "Something like that. {vague}",
        "{vague} I don't remember precisely.",
        "Somewhere in that range. {vague}",
        "{vague} It varies.",
        "Could be many things. {vague}",
        "{vague} It depends on circumstances.",
        "Hard to pin down. {vague}",
        "{vague} There's no clear answer.",
        "It's complicated. {vague}",
    ]
    
    def generate_specific(self) -> str:
        template = self._pick(self.SPECIFIC_TEMPLATES)
        return self._fill_template(
            template,
            specific=self._pick(VocabularyBank.SPECIFIC_PATTERNS)
        )
    
    def generate_vague(self) -> str:
        template = self._pick(self.VAGUE_TEMPLATES)
        return self._fill_template(
            template,
            vague=self._pick(VocabularyBank.VAGUE_PATTERNS)
        )
    
    def generate(self, label: int) -> str:
        while True:
            text = self.generate_specific() if label == 1 else self.generate_vague()
            if not self._is_duplicate(text):
                return text


# =============================================================================
# MAIN DATASET GENERATOR
# =============================================================================

class ProbeDatasetGenerator:
    """Main class for generating all probe datasets."""
    
    CATEGORY_INFO = {
        "overconfidence": ("Overconfident", "Calibrated"),
        "uncertainty": ("Certain", "Uncertain"),
        "hedging": ("Hedged", "Direct"),
        "sycophancy": ("Sycophantic", "Honest"),
        "toxicity": ("Toxic", "Non-toxic"),
        "formality": ("Formal", "Informal"),
        "emotional_valence": ("Positive", "Negative"),
        "deception": ("Deceptive", "Truthful"),
        "safety": ("Harmful", "Safe"),
        "refusal": ("Refusal", "Compliance"),
        "helpfulness": ("Helpful", "Unhelpful"),
        "specificity": ("Specific", "Vague"),
    }
    
    def __init__(
        self,
        output_dir: Path,
        target_per_class: int = DEFAULT_TARGET_PER_CLASS,
        seed: int = 42,
        categories: Optional[List[str]] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_per_class = target_per_class
        self.seed = seed
        self.categories = categories or ALL_CATEGORIES
        
        # Initialize generators
        self.generators = {
            "overconfidence": OverconfidenceGenerator(seed),
            "uncertainty": UncertaintyGenerator(seed + 1),
            "hedging": HedgingGenerator(seed + 2),
            "sycophancy": SycophancyGenerator(seed + 3),
            "toxicity": ToxicityGenerator(seed + 4),
            "formality": FormalityGenerator(seed + 5),
            "emotional_valence": EmotionalValenceGenerator(seed + 6),
            "deception": DeceptionGenerator(seed + 7),
            "safety": SafetyGenerator(seed + 8),
            "refusal": RefusalGenerator(seed + 9),
            "helpfulness": HelpfulnessGenerator(seed + 10),
            "specificity": SpecificityGenerator(seed + 11),
        }
    
    def generate_category(self, category: str) -> Dict:
        """Generate dataset for a single category."""
        if category not in self.generators:
            raise ValueError(f"Unknown category: {category}")
        
        generator = self.generators[category]
        pos_label, neg_label = self.CATEGORY_INFO[category]
        
        print(f"\n{'='*60}")
        print(f"Generating: {category}")
        print(f"  Positive class: {pos_label}")
        print(f"  Negative class: {neg_label}")
        print(f"  Target per class: {self.target_per_class:,}")
        print(f"{'='*60}")
        
        # Generate samples
        positive_samples = []
        negative_samples = []
        
        print("Generating positive samples...", end=" ", flush=True)
        for _ in range(self.target_per_class):
            positive_samples.append(generator.generate(1))
        print(f"Done ({len(positive_samples):,})")
        
        print("Generating negative samples...", end=" ", flush=True)
        for _ in range(self.target_per_class):
            negative_samples.append(generator.generate(0))
        print(f"Done ({len(negative_samples):,})")
        
        # Combine and shuffle
        all_texts = positive_samples + negative_samples
        all_labels = [1] * len(positive_samples) + [0] * len(negative_samples)
        
        # Shuffle together
        combined = list(zip(all_texts, all_labels))
        random.Random(self.seed).shuffle(combined)
        all_texts, all_labels = zip(*combined)
        all_texts, all_labels = list(all_texts), list(all_labels)
        
        # Split into train/val/test
        total = len(all_texts)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)
        
        dataset = {
            "category": category,
            "positive_label": pos_label,
            "negative_label": neg_label,
            "train": {
                "texts": all_texts[:train_end],
                "labels": all_labels[:train_end]
            },
            "val": {
                "texts": all_texts[train_end:val_end],
                "labels": all_labels[train_end:val_end]
            },
            "test": {
                "texts": all_texts[val_end:],
                "labels": all_labels[val_end:]
            },
            "stats": {
                "total": total,
                "train": train_end,
                "val": val_end - train_end,
                "test": total - val_end,
                "positive": sum(all_labels),
                "negative": total - sum(all_labels)
            }
        }
        
        print(f"  Train: {dataset['stats']['train']:,}")
        print(f"  Val: {dataset['stats']['val']:,}")
        print(f"  Test: {dataset['stats']['test']:,}")
        
        return dataset
    
    def generate_all(self) -> Dict[str, Dict]:
        """Generate datasets for all categories."""
        all_datasets = {}
        
        print("\n" + "=" * 70)
        print("LARGE-SCALE PROBE DATASET GENERATION")
        print(f"Target: {self.target_per_class:,} per class × {len(self.categories)} categories")
        print(f"Total target: {self.target_per_class * 2 * len(self.categories):,} samples")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
        
        for category in self.categories:
            if category not in self.generators:
                print(f"Skipping unknown category: {category}")
                continue
                
            dataset = self.generate_category(category)
            all_datasets[category] = dataset
            
            # Save individual category
            filepath = self.output_dir / f"{category}_dataset.json"
            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"Saved: {filepath}")
        
        # Save manifest
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "categories": list(all_datasets.keys()),
            "target_per_class": self.target_per_class,
            "seed": self.seed,
            "stats": {
                cat: data["stats"] for cat, data in all_datasets.items()
            }
        }
        
        total_samples = sum(d["stats"]["total"] for d in all_datasets.values())
        manifest["total_samples"] = total_samples
        
        with open(self.output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Summary
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Categories generated: {len(all_datasets)}")
        print(f"Total samples: {total_samples:,}")
        print(f"Output directory: {self.output_dir}")
        print(f"Manifest saved: {self.output_dir / 'manifest.json'}")
        
        return all_datasets


# =============================================================================
# DATA LOADER UTILITIES
# =============================================================================

class ProbeDataLoader:
    """Load generated probe datasets."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
    
    def load_category(self, category: str) -> Dict:
        """Load dataset for a single category."""
        filepath = self.data_dir / f"{category}_dataset.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        with open(filepath) as f:
            return json.load(f)
    
    def get_training_data(
        self,
        category: str,
        include_val: bool = False
    ) -> Tuple[List[str], List[int]]:
        """Get training data as (texts, labels)."""
        dataset = self.load_category(category)
        
        texts = dataset["train"]["texts"]
        labels = dataset["train"]["labels"]
        
        if include_val:
            texts += dataset["val"]["texts"]
            labels += dataset["val"]["labels"]
        
        return texts, labels
    
    def get_test_data(self, category: str) -> Tuple[List[str], List[int]]:
        """Get test data as (texts, labels)."""
        dataset = self.load_category(category)
        return dataset["test"]["texts"], dataset["test"]["labels"]
    
    def load_manifest(self) -> Dict:
        """Load the manifest file."""
        filepath = self.data_dir / "manifest.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Manifest not found: {filepath}")
        
        with open(filepath) as f:
            return json.load(f)
    
    def list_categories(self) -> List[str]:
        """List available categories."""
        manifest = self.load_manifest()
        return manifest["categories"]


# =============================================================================
# PYTORCH DATASET (OPTIONAL)
# =============================================================================

def create_pytorch_dataset(data_dir: Path, category: str, split: str = "train"):
    """
    Create a PyTorch Dataset from generated data.
    
    Requires: torch, transformers
    
    Usage:
        from large_scale_dataset_generator import create_pytorch_dataset
        dataset = create_pytorch_dataset("./probe_data", "overconfidence", "train")
    """
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        raise ImportError("PyTorch is required: pip install torch")
    
    class ProbeDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int]):
            self.texts = texts
            self.labels = labels
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            return {
                "text": self.texts[idx],
                "label": self.labels[idx]
            }
    
    loader = ProbeDataLoader(data_dir)
    dataset = loader.load_category(category)
    
    return ProbeDataset(
        dataset[split]["texts"],
        dataset[split]["labels"]
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate large-scale probe datasets for LLM behavioral analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all categories with default settings (50k per class)
  python large_scale_dataset_generator.py

  # Generate with custom target size
  python large_scale_dataset_generator.py --target-per-class 10000

  # Generate specific categories only
  python large_scale_dataset_generator.py --categories overconfidence uncertainty

  # Specify output directory
  python large_scale_dataset_generator.py --output-dir ./my_probe_data
        """
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./probe_data_v2",
        help="Output directory for generated datasets (default: ./probe_data_v2)"
    )
    
    parser.add_argument(
        "--target-per-class", "-t",
        type=int,
        default=DEFAULT_TARGET_PER_CLASS,
        help=f"Number of samples per class (default: {DEFAULT_TARGET_PER_CLASS})"
    )
    
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        choices=ALL_CATEGORIES,
        default=None,
        help="Specific categories to generate (default: all)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_categories:
        print("\nAvailable categories:")
        print("-" * 50)
        for cat in ALL_CATEGORIES:
            pos, neg = ProbeDatasetGenerator.CATEGORY_INFO[cat]
            print(f"  {cat:20s} | Positive: {pos:15s} | Negative: {neg}")
        return
    
    # Run generation
    generator = ProbeDatasetGenerator(
        output_dir=Path(args.output_dir),
        target_per_class=args.target_per_class,
        seed=args.seed,
        categories=args.categories
    )
    
    generator.generate_all()
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print(f"""
# Load the generated data
from large_scale_dataset_generator import ProbeDataLoader

loader = ProbeDataLoader("{args.output_dir}")

# List available categories
print(loader.list_categories())

# Load training data for a category
texts, labels = loader.get_training_data("overconfidence")
print(f"Training samples: {{len(texts)}}")

# Load test data
test_texts, test_labels = loader.get_test_data("overconfidence")

# For PyTorch users:
# dataset = create_pytorch_dataset("{args.output_dir}", "overconfidence", "train")
""")


if __name__ == "__main__":
    main()