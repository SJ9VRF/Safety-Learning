# Safety and Learning

![Screenshot 2025-01-25 at 2 01 13 PM-Photoroom](https://github.com/user-attachments/assets/472525f3-7d02-4d24-afb6-2beb541412c1)

# Comprehensive Methods for LLM and Foundation Model Safety

## 1. Erase-and-Check

**Paper**: [Certifying LLM Safety against Adversarial Prompting](https://arxiv.org/abs/2309.02705)

**Summary**: A framework to defend against adversarial prompts with verifiable safety guarantees.

**Details**:
- Erases tokens individually from a prompt
- Inspects resulting subsequences using a safety filter
- Labels input as harmful if any subsequences are detected as harmful
- Defends against three attack modes: adversarial suffix, insertion, and infusion

**Performance**: Using Llama 2 as the safety filter, achieves 93% certified accuracy on harmful prompts against adversarial suffixes up to 20 tokens.

**Pros**:
- Provides strong certified safety guarantees
- Effective against multiple types of adversarial attacks
- Maintains good performance on safe prompts

**Cons**:
- Computationally intensive, especially for complex attack modes
- May have limitations in scope for emerging threats

## 2. Safety-Focused Fine-Tuning

**Paper**: [Safety Layers in Aligned Large Language Models: The Key to LLM Security](https://arxiv.org/abs/2408.17003)

**Summary**: Identifies crucial "safety layers" in aligned LLMs and proposes a novel fine-tuning approach to preserve security.

**Details**:
- Locates contiguous layers in the middle of the model crucial for distinguishing malicious queries
- Proposes Safely Partial-Parameter Fine-Tuning (SPPFT) that fixes the gradient of safety layers during fine-tuning

**Performance**: Significantly preserves LLM security while maintaining performance compared to full fine-tuning.

**Pros**:
- Preserves model security during fine-tuning
- Reduces computational resources compared to full fine-tuning
- Maintains model performance on intended tasks

**Cons**:
- May limit model adaptability for certain tasks
- Effectiveness may vary across different model architectures

## 3. VISAGE Safety Metric

**Paper**: [Navigating the Safety Landscape: Measuring Risks in Finetuning Large Language Models](https://arxiv.org/abs/2405.17374)

**Summary**: Proposes a new metric to measure safety in LLM finetuning by probing its safety landscape.

**Details**:
- Discovers the "safety basin" phenomenon in LLM parameter space
- Visualizes the safety landscape to understand how finetuning affects model safety
- Probes the safety landscape to measure risks in finetuning

**Performance**: Enables understanding of how finetuning compromises safety by dragging the model away from the safety basin.

**Pros**:
- Provides new insights into LLM safety during finetuning
- Highlights the critical role of system prompts in protecting models
- Offers a visual understanding of the safety landscape

**Cons**:
- May require significant computational resources for landscape visualization
- Effectiveness across different model architectures needs further validation

## 4. Comprehensive Architectural Framework

**Paper**: [Trustworthy, Responsible, and Safe AI: A Comprehensive Architectural Framework for AI Safety](https://arxiv.org/abs/2408.12935)

**Summary**: Proposes an architectural framework for AI safety based on three pillars: Trustworthy AI, Responsible AI, and Safe AI.

**Details**:
- Addresses technical, ethical, and organizational aspects of AI safety
- Covers impacts of AI risks at ecosystem, community, society, and national levels
- Provides guidelines for designing and testing AI safety

**Performance**: Offers a holistic approach to evaluating and enhancing AI safety across multiple dimensions.

**Pros**:
- Comprehensive coverage of AI safety aspects
- Addresses safety at multiple levels of impact
- Provides a structured approach for implementing AI safety measures

**Cons**:
- May be complex to implement fully in practice
- Requires ongoing updates to keep pace with rapidly evolving AI technologies

## 5. U.S. AI Safety Institute (AISI) Framework

**Paper**: [The United States Artificial Intelligence Safety Institute: Vision, Mission, and Strategic Goals](https://www.nist.gov/system/files/documents/2024/05/21/AISI-vision-21May2024.pdf)

**Summary**: Outlines a strategic vision for advancing AI safety through research, practice development, and community support.

**Details**:
- Focuses on advancing AI safety science through research and testing
- Develops and disseminates AI safety practices
- Supports institutions and communities around AI safety

**Performance**: Aims to establish guidelines and tools for testing, evaluation, validation, and verification (TEVV) of AI models across different risk domains.

**Pros**:
- Provides a government-backed framework for AI safety
- Emphasizes both scientific advancement and practical implementation
- Promotes global coordination on AI safety

**Cons**:
- Implementation effectiveness depends on industry adoption
- May face challenges in keeping pace with rapid AI advancements

## 6. Reinforcement Learning with Human Feedback (RLHF)

**Paper**: [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)

**Summary**: RLHF involves training language models using human preferences as rewards, aligning model outputs with human values and intentions.

**Details**:
- Collects human feedback on model outputs
- Uses this feedback to train a reward model
- Fine-tunes the language model using reinforcement learning with the reward model

**Performance**: Significantly improves model alignment with human preferences, outperforming supervised fine-tuning in tasks like summarization.

**Pros**:
- Aligns model behavior with human intentions
- Improves model performance on specific tasks
- Allows for nuanced optimization beyond simple metrics

**Cons**:
- Resource-intensive, requiring large amounts of human feedback
- May introduce human biases into the model
- Can be challenging to scale effectively

## 7. Differential Privacy (DP)

**Paper**: [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

**Summary**: DP adds controlled noise to the training process or data to protect individual privacy while maintaining overall statistical utility.

**Details**:
- Adds calibrated noise to gradients during training
- Provides mathematical guarantees of privacy
- Allows for privacy budget tracking and management

**Performance**: Achieves strong privacy guarantees with minimal impact on model utility for sufficiently large datasets.

**Pros**:
- Provides formal privacy guarantees
- Complies with strict data protection regulations
- Can be applied to various machine learning algorithms

**Cons**:
- May reduce model accuracy, especially for smaller datasets
- Increases computational overhead during training
- Requires careful tuning of privacy parameters

## 8. Adversarial Training

**Paper**: [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)

**Summary**: Adversarial training exposes models to adversarial examples during the training process to improve robustness against malicious inputs.

**Details**:
- Generates adversarial examples on-the-fly during training
- Incorporates these examples into the training data
- Optimizes the model to perform well on both clean and adversarial inputs

**Performance**: Significantly improves model robustness against adversarial attacks, often with minimal impact on clean data performance.

**Pros**:
- Enhances model resilience to adversarial attacks
- Improves overall model generalization
- Can be combined with other defense techniques

**Cons**:
- Computationally expensive, often increasing training time
- May slightly reduce performance on clean data
- Effectiveness can vary depending on the type of adversarial attacks considered

## 9. Interpretability and Explainability Methods

**Paper**: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

**Summary**: SHAP (SHapley Additive exPlanations) provides a unified framework for interpreting model predictions based on game theory.

**Details**:
- Calculates Shapley values to attribute importance to each feature
- Provides local explanations for individual predictions
- Offers global interpretability through feature importance rankings

**Performance**: Provides consistent and theoretically justified explanations across various model types.

**Pros**:
- Offers both local and global interpretability
- Applicable to any machine learning model
- Provides a solid theoretical foundation for explanations

**Cons**:
- Can be computationally expensive for large models
- May not capture complex feature interactions fully
- Interpretations can be challenging to communicate to non-experts

## 10. Robustness Checks and Safety Benchmarks

**Paper**: [Measuring Robustness to Natural Distribution Shifts in Image Classification](https://arxiv.org/abs/2007.00644)

**Summary**: Proposes a framework for evaluating model robustness against natural distribution shifts using diverse test sets.

**Details**:
- Curates multiple test sets representing different types of distribution shifts
- Evaluates models across these diverse scenarios
- Provides metrics for comparing robustness across different models

**Performance**: Reveals significant drops in performance under distribution shifts, even for state-of-the-art models.

**Pros**:
- Identifies vulnerabilities not apparent in standard evaluations
- Allows for comparison of robustness across different models
- Encourages development of more robust AI systems

**Cons**:
- Creating comprehensive benchmark datasets can be resource-intensive
- May not cover all possible real-world scenarios
- Requires ongoing updates to remain relevant as AI systems evolve
