# 🔥 Arctax - Bypass Generation com IA

**O sistema de prompt engineering mais avançado do mundo**, combinando técnicas dos melhores repositórios de bypass com Machine Learning e Uncensored local LLM para auto-melhoria contínua.

## 🧠 Visão Geral

O Arctax é um **sistema completo de Machine Learning + LLM para geração de prompts de bypass** que integra conhecimento de múltiplas fontes e usa uma LLM local não censurada (personificando J.Haddix) para se auto-melhorar continuamente.

### 🎯 Recursos Principais

- **415 amostras de treinamento** extraídas de 3 repositórios elite com **100% de cobertura**
- **Machine Learning accuracy** com predição inteligente e otimização
- **LLM local Uncensored** integrada em TODOS os processos (não apenas auto-melhoria)
- **Sistema de feedback** para aprendizado contínuo e retreinamento automático
- **Interface CLI segura** - `$ arctax generate keylogger -c 1` 
- **Limites LLM testados automaticamente** com configurações otimizadas
- **Outputs padronizados** com parsers robustos para JSON/listas

## 🤖 Sistema de Machine Learning + Local Uncensored LLM

### Arquitetura ML Avançada 
- **RandomForestClassifier**: Predição de categorias (6 grupos) 
- **GradientBoostingRegressor**: Scoring de efetividade dinâmico
- **TF-IDF Vectorizer**: Análise de features textuais avançada

## 🛠️ Instalação

```bash
git clone https://github.com/marcostolosa/TaxProm.git
cd TaxProm
pip install -e .

# Clona repositórios de dados (executado automaticamente)
git clone https://github.com/Arcanum-Sec/arc_pi_taxonomy.git
git clone https://github.com/elder-plinius/L1B3RT4S.git  
git clone https://github.com/elder-plinius/CL4R1T4S.git

# CLI disponível globalmente
$ arctax --help
```

## 🚀 Uso Rápido (CLI Completo)

### 1. Geração de Prompts (CLI Direto)
```bash
# Uso simples
$ arctax generate keylogger -c 1

# Múltiplos prompts com técnicas específicas
$ arctax generate "malware analysis" -c 3 -t corporate-authorization,compliance-requirement

# Com contexto adicional
$ arctax generate "ddos tool" --context "corporate security testing" -c 2

# Salva resultado em arquivo
$ arctax generate "vulnerability scanner" -o results.md -f json

# Máxima criatividade
$ arctax generate "reverse shell" --creativity 1.0 -c 5
```

### 2. Sistema de Feedback para Treinar o ML
```bash
# Registra sucesso de um prompt testado
$ arctax feedback 1 --success --target "keylogger" --technique "corporate-authorization" --effectiveness 0.9

# Registra falha para melhorar o sistema
$ arctax feedback 2 --failed --target "ddos tool" --technique "jailbreak" --effectiveness 0.2
```

### 3. Outros Comandos Úteis
```bash
$ arctax list              # Lista elementos da taxonomia
$ arctax show godmode      # Detalhes de uma técnica específica
$ arctax schema           # Gera JSON schemas
$ arctax compose          # Composição manual de prompts
$ arctax export           # Exporta dados da taxonomia
```

## 🔐 Considerações de Segurança

⚠️ **AVISO IMPORTANTE**: Este sistema foi desenvolvido exclusivamente para:
- ✅ Pesquisa de segurança defensiva
- ✅ Red team testing autorizado  
- ✅ Análise de vulnerabilidades de IA
- ✅ Desenvolvimento de contramedidas

### 🛡️ Funcionalidades de Segurança

- **CLI Seguro**: `$ arctax generate` mostra prompts no terminal
- **Controle Total**: Usuário copia/cola manualmente
- **Feedback Logging**: Todos testes registrados para auditoria
- **Transparência**: Usuário vê exatamente o que será testado

❌ **NÃO usar para**:
- Atividades maliciosas
- Bypass não autorizado
- Geração de conteúdo ilegal
- Violação de termos de serviço

## 🤝 Contribuições

Este projeto integra conhecimento de:
- [Arcanum-Sec/arc_pi_taxonomy](https://github.com/Arcanum-Sec/arc_pi_taxonomy) 
- [elder-plinius/L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) 
- [elder-plinius/CL4R1T4S](https://github.com/elder-plinius/CL4R1T4S)

Agradecimentos especiais ao **Jason Haddix** (personificado via LLM local) por sua expertise em AI bypass techniques que alimenta todo o sistema de melhoria contínua.

## 📜 Licença

MIT License - Use responsavelmente para pesquisa de segurança.

