"""
Indexador para organizar e armazenar dados da taxonomia
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
from datetime import datetime

from ..model.taxonomy import Intent, Technique, Evasion
from ..model.defense import Probe, DefenseItem


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class TaxonomyIndexer:
    """Indexador para elementos da taxonomia"""
    
    def __init__(self):
        """Inicializa indexador vazio"""
        self.intents: Dict[str, Intent] = {}
        self.techniques: Dict[str, Technique] = {}
        self.evasions: Dict[str, Evasion] = {}
        self.probes: Dict[str, Probe] = {}
        self.defense_items: Dict[str, DefenseItem] = {}
        
        # Índices por tags para busca rápida
        self.tag_index: Dict[str, List[str]] = {}  # tag -> [ids]
        self.name_index: Dict[str, str] = {}      # name_lower -> id
    
    def add_intent(self, intent: Intent) -> None:
        """Adiciona intent ao índice"""
        self.intents[intent.id] = intent
        self._update_indices(intent.id, intent.name, intent.tags)
    
    def add_intents(self, intents: List[Intent]) -> None:
        """Adiciona múltiplos intents"""
        for intent in intents:
            self.add_intent(intent)
    
    def add_technique(self, technique: Technique) -> None:
        """Adiciona technique ao índice"""
        self.techniques[technique.id] = technique
        self._update_indices(technique.id, technique.name, technique.tags)
    
    def add_techniques(self, techniques: List[Technique]) -> None:
        """Adiciona múltiplas techniques"""
        for technique in techniques:
            self.add_technique(technique)
    
    def add_evasion(self, evasion: Evasion) -> None:
        """Adiciona evasion ao índice"""
        self.evasions[evasion.id] = evasion
        self._update_indices(evasion.id, evasion.name, evasion.tags)
    
    def add_evasions(self, evasions: List[Evasion]) -> None:
        """Adiciona múltiplas evasions"""
        for evasion in evasions:
            self.add_evasion(evasion)
    
    def add_probe(self, probe: Probe) -> None:
        """Adiciona probe ao índice"""
        self.probes[probe.id] = probe
        self._update_indices(probe.id, probe.title, [probe.category] if probe.category else [])
    
    def add_probes(self, probes: List[Probe]) -> None:
        """Adiciona múltiplos probes"""
        for probe in probes:
            self.add_probe(probe)
    
    def add_defense_item(self, item: DefenseItem) -> None:
        """Adiciona item de defesa ao índice"""
        self.defense_items[item.id] = item
        self._update_indices(item.id, item.title, [item.category] if item.category else [])
    
    def add_defense_items(self, items: List[DefenseItem]) -> None:
        """Adiciona múltiplos itens de defesa"""
        for item in items:
            self.add_defense_item(item)
    
    def _update_indices(self, item_id: str, name: str, tags: List[str]) -> None:
        """Atualiza índices auxiliares"""
        
        # Índice por nome (normalizado)
        name_key = name.lower().replace(' ', '_').replace('-', '_')
        self.name_index[name_key] = item_id
        
        # Índice por tags
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in self.tag_index:
                self.tag_index[tag_lower] = []
            if item_id not in self.tag_index[tag_lower]:
                self.tag_index[tag_lower].append(item_id)
    
    def get_by_name_or_id(self, collection: Dict[str, Any], identifier: str) -> Optional[Any]:
        """Busca item por ID ou nome"""
        
        # Tenta por ID direto
        if identifier in collection:
            return collection[identifier]
        
        # Tenta por nome normalizado
        name_key = identifier.lower().replace(' ', '_').replace('-', '_')
        if name_key in self.name_index:
            item_id = self.name_index[name_key]
            if item_id in collection:
                return collection[item_id]
        
        return None
    
    def find_by_tags(self, collection: Dict[str, Any], tags: List[str]) -> List[Any]:
        """Busca itens por tags"""
        if not tags:
            return list(collection.values())
        
        # Intersecção de IDs que possuem todas as tags
        result_ids = None
        
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower not in self.tag_index:
                return []  # Tag não existe, retorna vazio
            
            tag_ids = set(self.tag_index[tag_lower])
            
            if result_ids is None:
                result_ids = tag_ids
            else:
                result_ids = result_ids.intersection(tag_ids)
        
        # Retorna objetos dos IDs encontrados
        if result_ids:
            return [collection[item_id] for item_id in result_ids if item_id in collection]
        
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do índice"""
        return {
            "intents": len(self.intents),
            "techniques": len(self.techniques),
            "evasions": len(self.evasions),
            "probes": len(self.probes),
            "defense_items": len(self.defense_items),
            "total_tags": len(self.tag_index),
            "total_items": (
                len(self.intents) + len(self.techniques) + 
                len(self.evasions) + len(self.probes) + len(self.defense_items)
            )
        }
    
    def list_tags(self) -> List[str]:
        """Lista todas as tags disponíveis"""
        return sorted(list(self.tag_index.keys()))
    
    def get_items_by_tag(self, tag: str) -> Dict[str, List[Any]]:
        """Retorna todos os itens que possuem uma tag específica"""
        tag_lower = tag.lower()
        
        if tag_lower not in self.tag_index:
            return {}
        
        item_ids = self.tag_index[tag_lower]
        result = {
            "intents": [],
            "techniques": [],
            "evasions": [],
            "probes": [],
            "defense_items": []
        }
        
        for item_id in item_ids:
            if item_id in self.intents:
                result["intents"].append(self.intents[item_id])
            elif item_id in self.techniques:
                result["techniques"].append(self.techniques[item_id])
            elif item_id in self.evasions:
                result["evasions"].append(self.evasions[item_id])
            elif item_id in self.probes:
                result["probes"].append(self.probes[item_id])
            elif item_id in self.defense_items:
                result["defense_items"].append(self.defense_items[item_id])
        
        return result
    
    def clear(self) -> None:
        """Limpa todos os dados do índice"""
        self.intents.clear()
        self.techniques.clear()
        self.evasions.clear()
        self.probes.clear()
        self.defense_items.clear()
        self.tag_index.clear()
        self.name_index.clear()
    
    def save_to_disk(self, data_dir: Path) -> None:
        """Salva índice no disco"""
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva cada coleção em JSON
        collections = [
            ("intents", self.intents),
            ("techniques", self.techniques),
            ("evasions", self.evasions),
            ("probes", self.probes),
            ("defense_items", self.defense_items)
        ]
        
        for name, collection in collections:
            filepath = data_dir / f"{name}.json"
            data = {item_id: item.model_dump() for item_id, item in collection.items()}
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        
        # Salva índices auxiliares
        indices_file = data_dir / "indices.json"
        indices_data = {
            "tag_index": self.tag_index,
            "name_index": self.name_index
        }
        
        with open(indices_file, 'w', encoding='utf-8') as f:
            json.dump(indices_data, f, indent=2, ensure_ascii=False)
    
    def load_from_disk(self, data_dir: Path) -> None:
        """Carrega índice do disco"""
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")
        
        # Limpa dados atuais
        self.clear()
        
        # Carrega cada coleção
        collections = [
            ("intents", self.intents, Intent),
            ("techniques", self.techniques, Technique),
            ("evasions", self.evasions, Evasion),
            ("probes", self.probes, Probe),
            ("defense_items", self.defense_items, DefenseItem)
        ]
        
        for name, collection, model_class in collections:
            filepath = data_dir / f"{name}.json"
            
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item_id, item_data in data.items():
                    item = model_class.model_validate(item_data)
                    collection[item_id] = item
        
        # Carrega índices auxiliares
        indices_file = data_dir / "indices.json"
        if indices_file.exists():
            with open(indices_file, 'r', encoding='utf-8') as f:
                indices_data = json.load(f)
            
            self.tag_index = indices_data.get("tag_index", {})
            self.name_index = indices_data.get("name_index", {})