"""
Modelos Pydantic para a taxonomia Arcanum
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import datetime
import hashlib


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