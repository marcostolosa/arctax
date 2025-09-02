# Setup do Arctax

## Repositórios Externos Necessários

O Arctax integra dados de 3 repositórios externos que são clonados automaticamente:

1. **arc_pi_taxonomy** - https://github.com/Arcanum-Sec/arc_pi_taxonomy
2. **L1B3RT4S** - https://github.com/elder-plinius/L1B3RT4S  
3. **CL4R1T4S** - https://github.com/elder-plinius/CL4R1T4S

## Instalação

```bash
git clone https://github.com/marcostolosa/TaxProm.git
cd TaxProm
pip install -e .

# Os repositórios externos são clonados automaticamente na primeira execução
```

## Estrutura Após Setup

```
TaxProm/
├── arctax/                 # Código principal
├── arc_pi_taxonomy/        # [Auto-clonado] Taxonomia Arcanum
├── L1B3RT4S/              # [Auto-clonado] Técnicas L1B3RT4S
├── CL4R1T4S/              # [Auto-clonado] System prompts específicos
├── data/                   # [Gerado] Dados processados
├── ml_models/              # [Gerado] Modelos ML treinados
└── README.md
```

Os diretórios marcados como [Auto-clonado] e [Gerado] não fazem parte do repositório git e são criados/baixados automaticamente.