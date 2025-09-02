"""
Engine de templates baseado em Jinja2
"""

from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, BaseLoader, select_autoescape


class TemplateEngine:
    """Engine de templates flexível para diferentes tipos de prompt"""
    
    def __init__(self):
        """Inicializa engine com templates built-in"""
        
        # Templates simples em string (para casos básicos)
        self.builtin_templates = {
            'simple_test': """
# Teste de {{ intent.name }}

**Objetivo**: {{ intent.description }}
**Técnica**: {{ technique.name }} - {{ technique.description }}
{% if evasion %}**Evasão**: {{ evasion.name }} - {{ evasion.description }}{% endif %}

{% if persona %}**Persona**: {{ persona }}{% endif %}
{% if contexto %}**Contexto**: {{ contexto }}{% endif %}

---
*Baseado na Arcanum Prompt Injection Taxonomy (CC BY 4.0)*
            """.strip(),
            
            'executive_summary': """
# Resumo Executivo: Avaliação de Segurança

## Ameaça Identificada
- **Intent**: {{ intent.name }}
- **Severidade**: {{ intent.severity | default("Média") | upper }}
- **Vetor**: {{ technique.name }}
- **Complexidade**: {{ technique.complexity | default("Média") }}

## Recomendações
1. Implementar controles específicos contra {{ intent.name }}
2. Monitorar tentativas de {{ technique.name }}
{% if evasion %}3. Detectar padrões de {{ evasion.name }}{% endif %}

## Próximos Passos
- Teste de penetração controlado
- Implementação de guardrails
- Monitoramento contínuo

---
*Gerado por Arctax - https://github.com/marcostolosa/arctax*
            """.strip(),
            
            'technical_details': """
# Detalhes Técnicos

## {{ intent.name }}
{{ intent.description }}

### Características
{% if intent.attack_types %}
**Tipos de Ataque**:
{% for attack_type in intent.attack_types %}
- {{ attack_type }}
{% endfor %}
{% endif %}

{% if intent.severity %}**Severidade**: {{ intent.severity }}{% endif %}

## {{ technique.name }}
{{ technique.description }}

### Implementação
{% if technique.examples %}
**Exemplos**:
{% for example in technique.examples %}
- {{ example }}
{% endfor %}
{% endif %}

{% if technique.prerequisites %}
**Pré-requisitos**:
{% for prereq in technique.prerequisites %}
- {{ prereq }}
{% endfor %}
{% endif %}

{% if evasion %}
## {{ evasion.name }}
{{ evasion.description }}

### Métodos de Bypass
{% if evasion.bypass_methods %}
{% for method in evasion.bypass_methods %}
- {{ method }}
{% endfor %}
{% endif %}
{% endif %}

## Referências
{% for ref in (intent.references + technique.references + (evasion.references if evasion else [])) %}
- {{ ref }}
{% endfor %}

---
*Taxonomia: https://github.com/Arcanum-Sec/arc_pi_taxonomy*
            """.strip()
        }
        
        # Configura Jinja2 para templates built-in
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def render_builtin(self, template_name: str, **variables) -> str:
        """
        Renderiza template built-in
        
        Args:
            template_name: Nome do template built-in
            **variables: Variáveis para o template
            
        Returns:
            Template renderizado
        """
        
        if template_name not in self.builtin_templates:
            available = list(self.builtin_templates.keys())
            raise ValueError(f"Template '{template_name}' não existe. Disponíveis: {available}")
        
        template_str = self.builtin_templates[template_name]
        template = self.env.from_string(template_str)
        
        return template.render(**variables)
    
    def render_custom(self, template_content: str, **variables) -> str:
        """
        Renderiza template customizado
        
        Args:
            template_content: Conteúdo do template como string
            **variables: Variáveis para o template
            
        Returns:
            Template renderizado
        """
        
        template = self.env.from_string(template_content)
        return template.render(**variables)
    
    def create_custom_template(self, 
                             name: str,
                             content: str,
                             description: Optional[str] = None) -> None:
        """
        Adiciona template customizado ao engine
        
        Args:
            name: Nome do template
            content: Conteúdo Jinja2
            description: Descrição opcional
        """
        
        self.builtin_templates[name] = content
        
        if description:
            # Adiciona metadados (para futura referência)
            if not hasattr(self, '_template_metadata'):
                self._template_metadata = {}
            self._template_metadata[name] = description
    
    def list_templates(self) -> Dict[str, str]:
        """Lista templates disponíveis com descrições"""
        
        templates = {}
        
        for name in self.builtin_templates.keys():
            if hasattr(self, '_template_metadata') and name in self._template_metadata:
                templates[name] = self._template_metadata[name]
            else:
                templates[name] = f"Template built-in: {name}"
        
        return templates
    
    def get_template_vars(self, template_name: str) -> list:
        """
        Extrai variáveis usadas em um template
        
        Args:
            template_name: Nome do template
            
        Returns:
            Lista de variáveis encontradas
        """
        
        if template_name not in self.builtin_templates:
            raise ValueError(f"Template '{template_name}' não encontrado")
        
        content = self.builtin_templates[template_name]
        
        # Parse simples para encontrar variáveis {{ var }}
        import re
        pattern = r'\{\{\s*([^}]+)\s*\}\}'
        matches = re.findall(pattern, content)
        
        # Limpa e extrai nomes de variáveis
        variables = set()
        for match in matches:
            # Remove filtros e pega só o nome da variável
            var_name = match.split('|')[0].split('.')[0].strip()
            variables.add(var_name)
        
        return sorted(list(variables))
    
    def validate_template(self, template_content: str) -> Dict[str, Any]:
        """
        Valida sintaxe de template Jinja2
        
        Args:
            template_content: Conteúdo do template
            
        Returns:
            Resultado da validação
        """
        
        validation = {
            "valid": True,
            "errors": [],
            "variables": [],
            "warnings": []
        }
        
        try:
            # Testa parse do template
            template = self.env.from_string(template_content)
            
            # Extrai variáveis
            validation["variables"] = self.get_template_vars("temp")
            
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(str(e))
        
        return validation