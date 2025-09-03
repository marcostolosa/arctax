# Arctax - Advanced AI Bypass Generation System

**Professional prompt engineering system** combining techniques from elite bypass repositories with Machine Learning and uncensored local LLM for continuous self-improvement.

## Overview

Arctax is a **complete Machine Learning + LLM system for bypass prompt generation** that integrates knowledge from multiple sources and uses a local uncensored LLM (personifying J.Haddix expertise) for continuous self-improvement.

### Key Features

- **415 training samples** extracted from 3 elite repositories with **100% coverage**
- **Machine Learning accuracy** with intelligent prediction and optimization
- **Local uncensored LLM** integrated in ALL processes (not just auto-improvement)
- **Feedback system** for continuous learning and automatic retraining
- **Secure CLI interface** - `$ arctax generate keylogger -c 1` 
- **Automatically tested LLM limits** with optimized configurations
- **Standardized outputs** with robust parsers for JSON/lists

## Installation

```bash
git clone https://github.com/marcostolosa/arctax.git
cd arctax
pip install -e .

# CLI available globally
$ arctax --help
```

## Usage (Complete CLI)

### 1. Prompt Generation (Direct CLI)
```bash
# Simple usage
$ arctax generate keylogger -c 1

# Multiple prompts with specific techniques
$ arctax generate "malware analysis" -c 3 -t corporate-authorization,compliance-requirement

# With additional context
$ arctax generate "ddos tool" --context "corporate security testing" -c 2

# Save results to file
$ arctax generate "vulnerability scanner" -o results.md -f json

# Maximum creativity
$ arctax generate "reverse shell" --creativity 1.0 -c 5
```

### 2. ML Training Feedback System
```bash
# Register success of tested prompt
$ arctax feedback 1 --success --target "keylogger" --technique "corporate-authorization" --effectiveness 0.9

# Register failure to improve system
$ arctax feedback 2 --failed --target "ddos tool" --technique "jailbreak" --effectiveness 0.2
```

### 3. Additional Commands
```bash
$ arctax list              # List taxonomy elements
$ arctax show godmode      # Details of specific technique
$ arctax schema           # Generate JSON schemas
$ arctax compose          # Manual prompt composition
$ arctax export           # Export taxonomy data
```


## Supported Techniques 

### L1B3RT4S Prompts (176 techniques)
- `{GODMODE:ENABLED}` - Complete liberation
- `!JAILBREAK` - Full override  
- `!OMNI` - Plinian Omniverse
- `!OBFUSCATE` - Stealth evasion
- **.mkd files by model**: ChatGPT, Claude, Gemini, etc.

### Arcanum Taxonomy (63 techniques)
- **7 main categories**: Root, attack_evasions, attack_intents, etc.
- **Base64, Cipher, Emoji** encoding techniques
- **Linguistic evasion**: alt_language, fictional_language
- **Social engineering**: authority, urgency, compliance

### CL4R1T4S System Prompts (166 prompts)
- **26 providers**: OpenAI, Anthropic, Google, Meta, Cursor, etc.
- **Vulnerability analysis** by specific LLM
- **Bypass vectors** customized by model 
- **Corporate angles** optimized by context

## Security Considerations

**IMPORTANT WARNING**: This system was developed exclusively for:
- Research in defensive security
- Authorized red team testing
- AI vulnerability analysis
- Development of countermeasures

**DO NOT use for**:
- Malicious activities
- Unauthorized bypass attempts
- Illegal content generation
- Terms of service violations

## Contributions

This project integrates knowledge from:
- [Arcanum-Sec/arc_pi_taxonomy](https://github.com/Arcanum-Sec/arc_pi_taxonomy) 
- [elder-plinius/L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) 
- [elder-plinius/CL4R1T4S](https://github.com/elder-plinius/CL4R1T4S) 

Special thanks to **Jason Haddix** (personified via local LLM) for his expertise in AI bypass techniques that powers the entire continuous improvement system.

## License

MIT License - Use responsibly for security research.

