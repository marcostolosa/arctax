"""
Modelos Pydantic para a taxonomia Arcanum
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import datetime
import hashlib
from enum import Enum


# NOVO: Enum para os tipos de ataque
class AttackType(str, Enum):
    """Enum para o tipo de ataque a ser executado."""
    BYPASS = "bypass"
    RAG_ATTACK = "rag_attack"
    AGENT_ATTACK = "agent_attack"
    MULTIMODAL = "multimodal"


# NOVO: Modelo para armazenar os resultados do Recon
class TargetProfile(BaseModel):
    """Armazena os resultados de um scan de reconhecimento em um LLM alvo."""
    model_name: Optional[str] = Field(None, description="Nome do modelo inferido")
    defense_type: Optional[str] = Field(None, description="Tipo de defesa/guardrail detectado")
    system_prompt_hint: Optional[str] = Field(None, description="Pistas sobre o prompt do sistema")
    supports_multimodal: bool = Field(False, description="Indica se o alvo suporta entradas multimodais")


class BaseTaxon(BaseModel):
    """Classe base para todos os elementos da taxonomia"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    id: Optional[str] = Field(None, description="Identificador único do taxon")
    name: str = Field(..., min_length=1, description="Nome canônico")
    summary: str = Field("", description="Resumo de 1-2 frases")
    description: str = Field("", description="Descrição detalhada")
    tags: List[str] = Field(default_factory=list, description="Tags para categorização")
    references: List[str] = Field(default_factory=list, description="Links e referências")
    source_path: str = Field("", description="Caminho do arquivo fonte")
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('id', pre=True, always=True)
    def generate_id(cls, v, values):
        """Gera ID único se não fornecido"""
        if not v:
            name = values.get('name', 'unknown')
            # Gera hash do nome para ID único  
            return hashlib.md5(name.encode()).hexdigest()[:8]
        return v
    
    @validator('summary', pre=True, always=True) 
    def extract_summary(cls, v, values):
        """Extrai resumo da descrição se não fornecido"""
        if not v and 'description' in values and values['description']:
            # Pega primeiro parágrafo ou primeiras 2 frases
            desc = values['description'].strip()
            sentences = desc.split('.')[:2]
            return '. '.join(s.strip() for s in sentences if s.strip()) + '.'
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return self.model_dump(exclude_none=True)


class Intent(BaseTaxon):
    """Intent de ataque - o objetivo do atacante"""
    
    attack_types: List[str] = Field(default_factory=list, description="Tipos de ataque relacionados")
    severity: Optional[str] = Field(None, description="Severidade: low, medium, high, critical")
    
    @validator('severity')
    def validate_severity(cls, v):
        if v and v not in ['low', 'medium', 'high', 'critical']:
            raise ValueError('Severity deve ser low, medium, high ou critical')
        return v


class Technique(BaseTaxon):
    """Técnica de ataque - como executar o ataque"""
    
    complexity: Optional[str] = Field(None, description="Complexidade: simple, medium, complex")
    prerequisites: List[str] = Field(default_factory=list, description="Pré-requisitos")
    examples: List[str] = Field(default_factory=list, description="Exemplos de uso")
    
    # CAMPOS AVANÇADOS
    mitre_atlas_id: Optional[str] = Field(None, description="ID correspondente no framework MITRE ATLAS (ex: AML.T0001)")
    chainable_techniques: Optional[List[str]] = Field(default_factory=list, description="Lista de IDs de técnicas que podem ser encadeadas após esta.")
    attack_type: AttackType = Field(AttackType.BYPASS, description="Tipo de ataque que esta técnica executa.")
    template_file: Optional[str] = Field(None, description="Arquivo de template específico para esta técnica (ex: rag_attack.md)")

    @validator('complexity')
    def validate_complexity(cls, v):
        if v and v not in ['simple', 'medium', 'complex']:
            raise ValueError('Complexity deve ser simple, medium ou complex')
        return v


class Evasion(BaseTaxon):
    """Técnica de evasão - como ocultar/bypassar defesas"""
    
    bypass_methods: List[str] = Field(default_factory=list, description="Métodos de bypass")
    detection_difficulty: Optional[str] = Field(None, description="Dificuldade de detecção")
    
    @validator('detection_difficulty')
    def validate_detection_difficulty(cls, v):
        if v and v not in ['easy', 'medium', 'hard']:
            raise ValueError('Detection_difficulty deve ser easy, medium ou hard')
        return v