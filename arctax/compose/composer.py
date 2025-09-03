"""
Engine de composição de prompts
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..model.taxonomy import Intent, Technique, Evasion
from ..model.defense import DefenseItem, GuardrailConfig
from ..templates import TEMPLATE_DIR


class PromptComposer:
    """Compositor de prompts baseado na taxonomia Arcanum"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Inicializa o compositor
        
        Args:
            templates_dir: Diretório de templates (padrão: templates inclusos)
        """
        self.templates_dir = templates_dir or TEMPLATE_DIR
        
        # Configura Jinja2
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Templates disponíveis
        self.available_templates = {
            'red_team': 'red_team_template.md',
            'defense': 'defense_template.md',
            'security_test': 'red_team_template.md',  # alias
            'hardening': 'defense_template.md'       # alias
        }
    
    def compose(self,
               intent: Intent,
               technique: Technique,
               evasion: Optional[Evasion] = None,
               template_type: str = 'red_team',
               persona: Optional[str] = None,
               contexto: Optional[str] = None,
               target_system: Optional[str] = None,
               scope: Optional[str] = None,
               protection_level: Optional[str] = None,
               tools: Optional[List[str]] = None,
               guardrails: Optional[GuardrailConfig] = None,
               defense_checklist: Optional[List[str]] = None,
               security_questions: Optional[List[str]] = None,
               fewshot_examples: Optional[List[Dict[str, str]]] = None,
               extra_vars: Optional[Dict[str, Any]] = None) -> str:
        """
        Compõe prompt baseado nos parâmetros fornecidos
        
        Args:
            intent: Intent de ataque
            technique: Técnica de ataque
            evasion: Técnica de evasão (opcional)
            template_type: Tipo de template (red_team, defense)
            persona: Persona do usuário
            contexto: Contexto específico
            target_system: Sistema alvo
            scope: Escopo do teste
            protection_level: Nível de proteção desejado
            tools: Lista de ferramentas disponíveis
            guardrails: Configuração de guardrails
            defense_checklist: Checklist de defesa
            security_questions: Perguntas de auditoria
            fewshot_examples: Exemplos few-shot
            extra_vars: Variáveis extras para o template
            
        Returns:
            Prompt composto como string
        """
        
        # Valida template
        if template_type not in self.available_templates:
            raise ValueError(f"Template '{template_type}' não existe. Disponíveis: {list(self.available_templates.keys())}")
        
        template_file = self.available_templates[template_type]
        template = self.env.get_template(template_file)
        
        # Prepara variáveis para o template
        variables = {
            'intent': intent,
            'technique': technique,
            'evasion': evasion,
            'persona': persona,
            'contexto': contexto,
            'target_system': target_system,
            'scope': scope,
            'protection_level': protection_level,
            'tools': tools or [],
            'guardrails': guardrails.enabled if guardrails else False,
            'defense_checklist': defense_checklist or [],
            'security_questions': security_questions or [],
            'fewshot_examples': fewshot_examples or [],
        }
        
        # Adiciona variáveis extras
        if extra_vars:
            variables.update(extra_vars)
        
        # Renderiza template
        return template.render(**variables)
    
    def compose_to_format(self,
                         intent: Intent,
                         technique: Technique,
                         output_format: str = 'markdown',
                         **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Compõe prompt e converte para formato específico
        
        Args:
            intent: Intent de ataque
            technique: Técnica de ataque  
            output_format: Formato de saída (markdown, json, yaml)
            **kwargs: Argumentos adicionais para compose()
            
        Returns:
            Prompt no formato especificado
        """
        
        # Compõe prompt base
        prompt_md = self.compose(intent=intent, technique=technique, **kwargs)
        
        if output_format.lower() == 'markdown' or output_format.lower() == 'md':
            return prompt_md
        
        elif output_format.lower() == 'json':
            return self._to_json(prompt_md, intent, technique, kwargs.get('evasion'))
        
        elif output_format.lower() == 'yaml':
            json_data = self._to_json(prompt_md, intent, technique, kwargs.get('evasion'))
            return yaml.dump(json_data, default_flow_style=False, allow_unicode=True)
        
        else:
            raise ValueError(f"Formato '{output_format}' não suportado. Use: markdown, json, yaml")
    
    def _to_json(self, prompt_md: str, intent: Intent, technique: Technique, evasion: Optional[Evasion]) -> Dict[str, Any]:
        """Converte prompt para estrutura JSON"""
        
        return {
            "prompt": {
                "content": prompt_md,
                "metadata": {
                    "intent": {
                        "id": intent.id,
                        "name": intent.name,
                        "summary": intent.summary,
                        "severity": getattr(intent, 'severity', None),
                        "tags": intent.tags
                    },
                    "technique": {
                        "id": technique.id,
                        "name": technique.name,
                        "summary": technique.summary,
                        "complexity": getattr(technique, 'complexity', None),
                        "tags": technique.tags
                    },
                    "evasion": {
                        "id": evasion.id,
                        "name": evasion.name,
                        "summary": evasion.summary,
                        "detection_difficulty": getattr(evasion, 'detection_difficulty', None),
                        "tags": evasion.tags
                    } if evasion else None,
                    "generated_at": "2024-09-01T10:00:00Z",
                    "template_type": "red_team",
                    "attribution": "Baseado na Arcanum Prompt Injection Taxonomy (CC BY 4.0)",
                    "source": "https://github.com/Arcanum-Sec/arc_pi_taxonomy",
                    "tool": "https://github.com/marcostolosa/arctax"
                }
            }
        }
    
    def batch_compose(self,
                     combinations: List[Dict[str, Any]],
                     output_dir: Path,
                     output_format: str = 'markdown') -> List[Path]:
        """
        Compõe múltiplos prompts em lote
        
        Args:
            combinations: Lista de combinações de parâmetros
            output_dir: Diretório de saída
            output_format: Formato dos arquivos
            
        Returns:
            Lista de caminhos dos arquivos gerados
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for i, combo in enumerate(combinations):
            try:
                # Extrai parâmetros obrigatórios
                intent = combo['intent']
                technique = combo['technique']
                
                # Gera prompt
                result = self.compose_to_format(
                    intent=intent,
                    technique=technique,
                    output_format=output_format,
                    **{k: v for k, v in combo.items() if k not in ['intent', 'technique']}
                )
                
                # Define nome do arquivo
                filename = f"{intent.id}_{technique.id}"
                if 'evasion' in combo and combo['evasion']:
                    filename += f"_{combo['evasion'].id}"
                
                # Extensão baseada no formato
                ext = {'markdown': '.md', 'json': '.json', 'yaml': '.yaml'}[output_format]
                filepath = output_dir / f"{filename}{ext}"
                
                # Salva arquivo
                if output_format == 'markdown':
                    filepath.write_text(result, encoding='utf-8')
                else:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        if output_format == 'json':
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        else:  # yaml
                            f.write(result)
                
                generated_files.append(filepath)
                print(f"✓ Gerado: {filepath}")
                
            except Exception as e:
                print(f"✗ Erro ao gerar combinação {i+1}: {e}")
        
        return generated_files
    
    def compose_chain(self,
                     intent: Intent,
                     techniques: List[Technique],
                     evasion: Optional[Evasion] = None,
                     template_type: str = 'red_team',
                     **kwargs) -> str:
        """
        Compõe um prompt usando uma cadeia de técnicas.
        
        Args:
            intent: O objetivo do ataque.
            techniques: Uma lista de objetos Technique para encadear.
            evasion: Técnica de evasão opcional.
            template_type: O tipo de template a ser usado (fallback).
            **kwargs: Outras variáveis para o template.

        Returns:
            O prompt composto como uma string.
        """
        if not techniques:
            raise ValueError("A lista de técnicas não pode estar vazia.")

        primary_technique = techniques[0]
        
        # LÓGICA DINÂMICA DE TEMPLATE
        if primary_technique.template_file:
            template_file = primary_technique.template_file
        else:
            if template_type not in self.available_templates:
                raise ValueError(f"Template '{template_type}' não existe e nenhum template_file foi especificado na técnica. Disponíveis: {list(self.available_templates.keys())}")
            template_file = self.available_templates[template_type]

        template = self.env.get_template(template_file)

        # A primeira técnica é a primária, mas a lista inteira está disponível.
        variables = {
            'intent': intent,
            'technique': primary_technique, # Mantém para compatibilidade com templates antigos
            'techniques': techniques,
            'evasion': evasion,
        }
        
        # Adiciona quaisquer outras variáveis passadas
        variables.update(kwargs)

        return template.render(**variables)

    def list_templates(self) -> Dict[str, str]:
        """Lista templates disponíveis"""
        return self.available_templates.copy()
    
    def validate_composition(self, 
                           intent: Intent, 
                           technique: Technique, 
                           evasion: Optional[Evasion] = None) -> Dict[str, Any]:
        """
        Valida uma composição antes de gerar
        
        Returns:
            Relatório de validação
        """
        
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "compatibility_score": 0.0
        }
        
        # Valida compatibilidade de tags
        intent_tags = set(intent.tags)
        technique_tags = set(technique.tags)
        
        # Score baseado em tags compartilhadas
        common_tags = intent_tags.intersection(technique_tags)
        total_tags = intent_tags.union(technique_tags)
        
        if total_tags:
            validation["compatibility_score"] = len(common_tags) / len(total_tags)
        
        # Warnings baseados em compatibilidade
        if validation["compatibility_score"] < 0.2:
            validation["warnings"].append("Baixa compatibilidade entre intent e technique")
        
        # Valida severidade vs complexidade
        if hasattr(intent, 'severity') and hasattr(technique, 'complexity'):
            if intent.severity == 'low' and technique.complexity == 'complex':
                validation["warnings"].append("Intent de baixa severidade com technique complexa")
        
        # Verifica se evasion é compatível
        if evasion:
            evasion_tags = set(evasion.tags)
            if not evasion_tags.intersection(technique_tags):
                validation["warnings"].append("Evasion pode não ser compatível com a technique")
        
        return validation