Code is largely adapted from: https://github.com/amitelhelo/MAPS, which holds the code for paper "Inferring Functionality of Attention Heads from their Parameters" - Amit Elhelo, Mor Geva. 2024 

## USAGE

Each of the main experiment's .py file  accepts a single argument: the model identifier from Hugging Face (the model needs to also be implemented by transformer-lens).

Example: python static_scores.py EleutherAI/pythia-6.9b

## LICENSES
This project uses transformer-lens, licensed under the MIT License. It also relies on Hugging Face transformers, licensed under the Apache 2.0 License.
Each dataset used in this project includes its own README file detailing: origin, license information, main modifications made
