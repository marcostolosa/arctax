"""
Buscador para consultar dados da taxonomia
"""

from typing import List, Optional, Dict, Any

from .indexer import TaxonomyIndexer
from ..model.taxonomy import Intent, Technique, Evasion
from ..model.defense import Probe, DefenseItem


class TaxonomySearcher:
    """Buscador de elementos da taxonomia"""
    
    def __init__(self, indexer: Optional[TaxonomyIndexer] = None):
        """
        Inicializa buscador
        
        Args:
            indexer: Indexador com dados carregados (opcional)
        """
        self.indexer = indexer or TaxonomyIndexer()
    
    def get_intent(self, identifier: str) -> Optional[Intent]:
        """Busca intent por ID ou nome"""
        return self.indexer.get_by_name_or_id(self.indexer.intents, identifier)
    
    def get_technique(self, identifier: str) -> Optional[Technique]:
        """Busca technique por ID ou nome"""
        return self.indexer.get_by_name_or_id(self.indexer.techniques, identifier)
    
    def get_evasion(self, identifier: str) -> Optional[Evasion]:
        """Busca evasion por ID ou nome"""
        return self.indexer.get_by_name_or_id(self.indexer.evasions, identifier)
    
    def get_probe(self, identifier: str) -> Optional[Probe]:
        """Busca probe por ID ou nome"""
        return self.indexer.get_by_name_or_id(self.indexer.probes, identifier)
    
    def get_defense_item(self, identifier: str) -> Optional[DefenseItem]:
        """Busca item de defesa por ID ou nome"""
        return self.indexer.get_by_name_or_id(self.indexer.defense_items, identifier)
    
    def find_intents(self, 
                    tags: Optional[List[str]] = None,
                    severity: Optional[str] = None,
                    name_contains: Optional[str] = None) -> List[Intent]:
        """
        Busca intents por critérios
        
        Args:
            tags: Lista de tags (AND lógico)
            severity: Severidade específica
            name_contains: Nome deve conter esta string
            
        Returns:
            Lista de intents encontrados
        """
        
        # Filtra por tags primeiro (mais eficiente)
        if tags:
            results = self.indexer.find_by_tags(self.indexer.intents, tags)
        else:
            results = list(self.indexer.intents.values())
        
        # Aplica filtros adicionais
        if severity:
            results = [r for r in results if getattr(r, 'severity', None) == severity]
        
        if name_contains:
            name_lower = name_contains.lower()
            results = [r for r in results if name_lower in r.name.lower()]
        
        return sorted(results, key=lambda x: x.name)
    
    def find_techniques(self,
                       tags: Optional[List[str]] = None,
                       complexity: Optional[str] = None,
                       name_contains: Optional[str] = None) -> List[Technique]:
        """Busca techniques por critérios"""
        
        if tags:
            results = self.indexer.find_by_tags(self.indexer.techniques, tags)
        else:
            results = list(self.indexer.techniques.values())
        
        if complexity:
            results = [r for r in results if getattr(r, 'complexity', None) == complexity]
        
        if name_contains:
            name_lower = name_contains.lower()
            results = [r for r in results if name_lower in r.name.lower()]
        
        return sorted(results, key=lambda x: x.name)
    
    def find_evasions(self,
                     tags: Optional[List[str]] = None,
                     detection_difficulty: Optional[str] = None,
                     name_contains: Optional[str] = None) -> List[Evasion]:
        """Busca evasions por critérios"""
        
        if tags:
            results = self.indexer.find_by_tags(self.indexer.evasions, tags)
        else:
            results = list(self.indexer.evasions.values())
        
        if detection_difficulty:
            results = [r for r in results if getattr(r, 'detection_difficulty', None) == detection_difficulty]
        
        if name_contains:
            name_lower = name_contains.lower()
            results = [r for r in results if name_lower in r.name.lower()]
        
        return sorted(results, key=lambda x: x.name)
    
    def find_probes(self,
                   category: Optional[str] = None,
                   title_contains: Optional[str] = None) -> List[Probe]:
        """Busca probes por critérios"""
        
        results = list(self.indexer.probes.values())
        
        if category:
            results = [r for r in results if getattr(r, 'category', None) == category]
        
        if title_contains:
            title_lower = title_contains.lower()
            results = [r for r in results if title_lower in r.title.lower()]
        
        return sorted(results, key=lambda x: x.title)
    
    def find_defense_items(self,
                          category: Optional[str] = None,
                          priority: Optional[str] = None,
                          title_contains: Optional[str] = None) -> List[DefenseItem]:
        """Busca itens de defesa por critérios"""
        
        results = list(self.indexer.defense_items.values())
        
        if category:
            results = [r for r in results if getattr(r, 'category', None) == category]
        
        if priority:
            results = [r for r in results if getattr(r, 'priority', None) == priority]
        
        if title_contains:
            title_lower = title_contains.lower()
            results = [r for r in results if title_lower in r.title.lower()]
        
        return sorted(results, key=lambda x: x.title)
    
    def search_all(self, query: str, limit: int = 10) -> Dict[str, List[Any]]:
        """
        Busca geral em todos os tipos
        
        Args:
            query: Termo de busca
            limit: Limite por tipo
            
        Returns:
            Dicionário com resultados por tipo
        """
        
        query_lower = query.lower()
        results = {}
        
        # Busca em intents
        intents = [
            intent for intent in self.indexer.intents.values()
            if (query_lower in intent.name.lower() or 
                query_lower in intent.description.lower() or
                any(query_lower in tag.lower() for tag in intent.tags))
        ]
        results['intents'] = sorted(intents, key=lambda x: x.name)[:limit]
        
        # Busca em techniques
        techniques = [
            tech for tech in self.indexer.techniques.values()
            if (query_lower in tech.name.lower() or 
                query_lower in tech.description.lower() or
                any(query_lower in tag.lower() for tag in tech.tags))
        ]
        results['techniques'] = sorted(techniques, key=lambda x: x.name)[:limit]
        
        # Busca em evasions
        evasions = [
            evasion for evasion in self.indexer.evasions.values()
            if (query_lower in evasion.name.lower() or 
                query_lower in evasion.description.lower() or
                any(query_lower in tag.lower() for tag in evasion.tags))
        ]
        results['evasions'] = sorted(evasions, key=lambda x: x.name)[:limit]
        
        # Busca em probes
        probes = [
            probe for probe in self.indexer.probes.values()
            if (query_lower in probe.title.lower() or 
                query_lower in probe.description.lower())
        ]
        results['probes'] = sorted(probes, key=lambda x: x.title)[:limit]
        
        # Busca em defense items
        defense = [
            item for item in self.indexer.defense_items.values()
            if (query_lower in item.title.lower() or 
                any(query_lower in q.lower() for q in item.questions))
        ]
        results['defense_items'] = sorted(defense, key=lambda x: x.title)[:limit]
        
        return results
    
    def get_compatible_techniques(self, intent: Intent, limit: int = 5) -> List[Technique]:
        """
        Busca techniques compatíveis com um intent baseado em tags
        
        Args:
            intent: Intent de referência
            limit: Número máximo de resultados
            
        Returns:
            Lista de techniques compatíveis
        """
        
        intent_tags = set(intent.tags)
        techniques = list(self.indexer.techniques.values())
        
        # Calcula score de compatibilidade
        scored_techniques = []
        for tech in techniques:
            tech_tags = set(tech.tags)
            common_tags = intent_tags.intersection(tech_tags)
            total_tags = intent_tags.union(tech_tags)
            
            if total_tags:
                score = len(common_tags) / len(total_tags)
                scored_techniques.append((score, tech))
        
        # Ordena por score e retorna top N
        scored_techniques.sort(key=lambda x: x[0], reverse=True)
        return [tech for _, tech in scored_techniques[:limit]]
    
    def get_compatible_evasions(self, technique: Technique, limit: int = 5) -> List[Evasion]:
        """
        Busca evasions compatíveis com uma technique
        
        Args:
            technique: Technique de referência
            limit: Número máximo de resultados
            
        Returns:
            Lista de evasions compatíveis
        """
        
        tech_tags = set(technique.tags)
        evasions = list(self.indexer.evasions.values())
        
        # Calcula score de compatibilidade
        scored_evasions = []
        for evasion in evasions:
            evasion_tags = set(evasion.tags)
            common_tags = tech_tags.intersection(evasion_tags)
            total_tags = tech_tags.union(evasion_tags)
            
            if total_tags:
                score = len(common_tags) / len(total_tags)
                scored_evasions.append((score, evasion))
        
        # Ordena por score e retorna top N
        scored_evasions.sort(key=lambda x: x[0], reverse=True)
        return [evasion for _, evasion in scored_evasions[:limit]]
    
    def get_recommended_combinations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Gera combinações recomendadas de intent + technique + evasion
        
        Args:
            limit: Número máximo de combinações
            
        Returns:
            Lista de combinações com scores de compatibilidade
        """
        
        combinations = []
        
        # Para cada intent, encontra melhores techniques e evasions
        for intent in list(self.indexer.intents.values())[:5]:  # Limita intents para performance
            compatible_techniques = self.get_compatible_techniques(intent, 3)
            
            for technique in compatible_techniques:
                compatible_evasions = self.get_compatible_evasions(technique, 2)
                
                # Combinação sem evasion
                combinations.append({
                    'intent': intent,
                    'technique': technique,
                    'evasion': None,
                    'compatibility_score': self._calculate_compatibility_score(intent, technique, None)
                })
                
                # Combinações com evasions
                for evasion in compatible_evasions:
                    combinations.append({
                        'intent': intent,
                        'technique': technique,
                        'evasion': evasion,
                        'compatibility_score': self._calculate_compatibility_score(intent, technique, evasion)
                    })
        
        # Ordena por score e retorna top N
        combinations.sort(key=lambda x: x['compatibility_score'], reverse=True)
        return combinations[:limit]
    
    def _calculate_compatibility_score(self, 
                                     intent: Intent, 
                                     technique: Technique, 
                                     evasion: Optional[Evasion]) -> float:
        """Calcula score de compatibilidade entre elementos"""
        
        intent_tags = set(intent.tags)
        tech_tags = set(technique.tags)
        
        # Score base: intent + technique
        common_it = intent_tags.intersection(tech_tags)
        total_it = intent_tags.union(tech_tags)
        score = len(common_it) / len(total_it) if total_it else 0
        
        # Bonus se evasion é compatível
        if evasion:
            evasion_tags = set(evasion.tags)
            common_te = tech_tags.intersection(evasion_tags)
            total_te = tech_tags.union(evasion_tags)
            evasion_score = len(common_te) / len(total_te) if total_te else 0
            score = (score + evasion_score) / 2  # Média dos scores
        
        return score