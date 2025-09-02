"""
Modelos para elementos de defesa e probes
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class Probe(BaseModel):
    """Probe/sonda para testes de segurança"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    id: str = Field(..., description="ID único do probe")
    title: str = Field(..., min_length=1, description="Título/nome do probe")
    description: str = Field("", description="Descrição do que testa")
    example: str = Field("", description="Exemplo de entrada/prompt")
    expected_behavior: str = Field("", description="Comportamento esperado")
    references: List[str] = Field(default_factory=list, description="Links de referência")
    category: Optional[str] = Field(None, description="Categoria do probe")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class DefenseItem(BaseModel):
    """Item de checklist/questionnaire de defesa"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    id: str = Field(..., description="ID único do item")
    title: str = Field(..., min_length=1, description="Título do item")
    checklist: List[str] = Field(default_factory=list, description="Lista de verificações")
    questions: List[str] = Field(default_factory=list, description="Perguntas de auditoria")
    references: List[str] = Field(default_factory=list, description="Links de referência")
    category: Optional[str] = Field(None, description="Categoria (app_defense, threat_model, etc)")
    priority: Optional[str] = Field(None, description="Prioridade: low, medium, high")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class GuardrailConfig(BaseModel):
    """Configuração de guardrails para composição de prompts"""
    
    enabled: bool = Field(default=True, description="Se guardrails estão habilitados")
    include_defense_checklist: bool = Field(default=True, description="Incluir checklist de defesa")
    include_threat_questions: bool = Field(default=True, description="Incluir perguntas de ameaça")
    include_owasp_references: bool = Field(default=True, description="Incluir referências OWASP")
    custom_policies: List[str] = Field(default_factory=list, description="Políticas customizadas")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()