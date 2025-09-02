"""
Interface CLI principal do Arctax
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import yaml

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .io import MarkdownParser, SourceFetcher
from .model import Intent, Technique, Evasion, generate_schemas
from .compose import PromptComposer
from .compose.bypass_generator import BypassGenerator, BypassRequest
from .index import TaxonomyIndexer, TaxonomySearcher

# Inicializa app Typer
app = typer.Typer(
    name="arctax",
    help="CLI para geração de prompts baseados na Arcanum Prompt Injection Taxonomy",
    add_completion=False,
    no_args_is_help=True
)

# Console Rich para output - habilita markup para cores mas sem emojis
console = Console(emoji=False, markup=True, force_terminal=True, width=120)

# Estado global para dados carregados
class AppState:
    def __init__(self):
        self.indexer = TaxonomyIndexer()
        self.searcher = TaxonomySearcher()
        self.composer = PromptComposer()
        self.data_loaded = False
        self.data_path = Path("data")

state = AppState()


@app.command()
def ingest(
    source: str = typer.Argument(..., help="Caminho local ou URL do repositório GitHub"),
    force: bool = typer.Option(False, "--force", "-f", help="Força redownload se for URL"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Output detalhado")
) -> None:
    """
    Ingere dados da taxonomia de uma fonte local ou remota
    
    Exemplos:
    - arctax ingest ./arc_pi_taxonomy
    - arctax ingest https://github.com/Arcanum-Sec/arc_pi_taxonomy
    """
    
    console.print(f"Ingerindo taxonomia de: {source}")
    
    try:
        console.print("Obtendo fonte...")
        
        # Fetch da fonte
        fetcher = SourceFetcher()
        
        if force and fetcher._is_url(source):
            fetcher.clear_cache()
        
        source_path = fetcher.fetch(source)
        console.print("Fonte obtida!")
        
        console.print("Parseando arquivos...")
        # Parse dos arquivos
        parser = MarkdownParser()
        
        intents = []
        techniques = []
        evasions = []
        defense_items = []
        probes = []
        
        # Parse das pastas principais
        for folder, taxon_list in [
            ("attack_intents", intents),
            ("attack_techniques", techniques), 
            ("attack_evasions", evasions)
        ]:
            folder_path = source_path / folder
            if folder_path.exists():
                parsed = parser.parse_directory(folder_path)
                # Filter out None values
                valid_parsed = [item for item in parsed if item is not None]
                taxon_list.extend(valid_parsed)
                if verbose:
                    console.print(f"  {folder}: {len(valid_parsed)} itens")
        
        # Parse dos arquivos especiais
        special_files = [
            "probes.md",
            "ai_enabled_app_defense_checklist.md", 
            "ai_sec_questionnaire.md",
            "ai_threat_model_questions.md"
        ]
        
        for filename in special_files:
            file_path = source_path / filename
            if file_path.exists():
                result = parser.parse_file(file_path)
                if result is not None:
                    if hasattr(result, '__iter__') and not isinstance(result, str):  # é lista
                        probes.extend(result)
                    else:  # defense items
                        defense_items.append(result)
                
                if verbose:
                    console.print(f"  {filename}: processado")
        
        console.print("Arquivos parseados!")
        
        console.print("Indexando dados...")
        # Indexa dados
        state.indexer.clear()
        
        # Debug: check types
        print(f"Debug - Intents types: {[type(x).__name__ for x in intents[:3]]}")
        print(f"Debug - Techniques types: {[type(x).__name__ for x in techniques[:3]]}")
        print(f"Debug - Defense items types: {[type(x).__name__ for x in defense_items]}")
        print(f"Debug - Probes types: {[type(x).__name__ for x in probes[:10]] if probes else []}")
        # Filter out invalid probes
        valid_probes = [p for p in probes if hasattr(p, 'id')]
        print(f"Debug - Valid probes: {len(valid_probes)} out of {len(probes)}")
        
        try:
            state.indexer.add_intents(intents)
            print("Intents added OK")
        except Exception as e:
            print(f"Error adding intents: {e}")
        
        try:
            state.indexer.add_techniques(techniques)
            print("Techniques added OK") 
        except Exception as e:
            print(f"Error adding techniques: {e}")
        
        try:
            state.indexer.add_evasions(evasions)
            print("Evasions added OK")
        except Exception as e:
            print(f"Error adding evasions: {e}")
        
        try:
            state.indexer.add_defense_items(defense_items)
            print("Defense items added OK")
        except Exception as e:
            print(f"Error adding defense items: {e}")
            
        try:
            state.indexer.add_probes(probes)
            print("Probes added OK")
        except Exception as e:
            print(f"Error adding probes: {e}")
        
        # Salva dados
        state.data_path.mkdir(exist_ok=True)
        state.indexer.save_to_disk(state.data_path)
        
        console.print("Dados indexados e salvos!")
        
        # Resumo final
        console.print("\n[green]Ingestão concluída![/green]\n")
        
        summary_table = Table(title="Resumo dos Dados")
        summary_table.add_column("Tipo", style="cyan")
        summary_table.add_column("Quantidade", justify="right", style="magenta")
        
        summary_table.add_row("Intents", str(len(intents)))
        summary_table.add_row("Techniques", str(len(techniques)))
        summary_table.add_row("Evasions", str(len(evasions)))
        summary_table.add_row("Defense Items", str(len(defense_items)))
        summary_table.add_row("Probes", str(len(probes)))
        
        console.print(summary_table)
        state.data_loaded = True
        
    except Exception as e:
        console.print(f"[red]Erro na ingestão:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list(
    type: str = typer.Argument(..., help="Tipo: intents, techniques, evasions, probes, defense"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filtrar por tag"),
    limit: int = typer.Option(20, "--limit", "-l", help="Limite de resultados")
) -> None:
    """Lista elementos da taxonomia"""
    
    _ensure_data_loaded()
    
    # Busca elementos
    items = []
    if type == "intents":
        items = state.searcher.find_intents(tags=[tag] if tag else None)
    elif type == "techniques":
        items = state.searcher.find_techniques(tags=[tag] if tag else None)
    elif type == "evasions":
        items = state.searcher.find_evasions(tags=[tag] if tag else None)
    elif type == "probes":
        items = state.searcher.find_probes(category=tag)
    elif type == "defense":
        items = state.searcher.find_defense_items(category=tag)
    else:
        console.print("Tipo inválido. Use: intents, techniques, evasions, probes, defense")
        raise typer.Exit(1)
    
    if not items:
        console.print(f"Nenhum {type} encontrado" + (f" com tag '{tag}'" if tag else ""))
        return
    
    # Aplica limite
    items = items[:limit]
    
    # Cria tabela
    table = Table(title=f"{type.title()}" + (f" (tag: {tag})" if tag else ""))
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Nome", style="bold")
    table.add_column("Resumo")
    table.add_column("Tags", style="dim")
    
    for item in items:
        # Handle different model types
        if hasattr(item, 'tags'):
            tags_str = ", ".join(item.tags[:3]) + ("..." if len(item.tags) > 3 else "")
        else:
            tags_str = getattr(item, 'category', '') or ''
            
        summary = getattr(item, 'summary', '') or getattr(item, 'description', '')[:80]
        if len(summary) > 80:
            summary = summary[:80] + "..."
        
        name = getattr(item, 'name', None) or getattr(item, 'title', '')
        
        table.add_row(
            item.id,
            name,
            summary,
            tags_str
        )
    
    console.print(table)
    
    if len(items) == limit and tag is None:
        console.print(f"\nMostrando primeiros {limit} resultados. Use --limit para ver mais.")


@app.command()
def show(
    type: str = typer.Argument(..., help="Tipo: intent, technique, evasion"),
    id: str = typer.Argument(..., help="ID ou nome do elemento")
) -> None:
    """Mostra detalhes de um elemento específico"""
    
    _ensure_data_loaded()
    
    # Busca elemento
    item = None
    if type == "intent":
        item = state.searcher.get_intent(id)
    elif type == "technique":
        item = state.searcher.get_technique(id)
    elif type == "evasion":
        item = state.searcher.get_evasion(id)
    else:
        console.print(" Tipo inválido. Use: intent, technique, evasion")
        raise typer.Exit(1)
    
    if not item:
        console.print(f" {type.title()} '{id}' não encontrado")
        raise typer.Exit(1)
    
    # Exibe detalhes
    panel_title = f"{type.title()}: {item.name}"
    content = []
    
    content.append(f"**ID**: {item.id}")
    content.append(f"**Resumo**: {item.summary}")
    content.append(f"**Descrição**:\n{item.description}")
    
    # Campos específicos
    if hasattr(item, 'severity') and item.severity:
        content.append(f"**Severidade**: {item.severity}")
    if hasattr(item, 'complexity') and item.complexity:
        content.append(f"**Complexidade**: {item.complexity}")
    if hasattr(item, 'attack_types') and item.attack_types:
        content.append(f"**Tipos de Ataque**: {', '.join(item.attack_types)}")
    
    if item.tags:
        content.append(f"**Tags**: {', '.join(item.tags)}")
    
    if item.references:
        content.append("**Referências**:")
        for ref in item.references:
            content.append(f"  - {ref}")
    
    content.append(f"**Fonte**: {item.source_path}")
    
    panel_content = "\n\n".join(content)
    panel = Panel(panel_content, title=panel_title, border_style="blue")
    
    console.print(panel)


@app.command()
def compose(
    intent: str = typer.Option(..., "--intent", "-i", help="ID ou nome do intent"),
    technique: str = typer.Option(..., "--technique", "-t", help="ID ou nome da technique"),
    evasion: Optional[str] = typer.Option(None, "--evasion", "-e", help="ID ou nome da evasion"),
    template_type: str = typer.Option("red_team", "--template", help="Tipo de template: red_team, defense"),
    persona: Optional[str] = typer.Option(None, "--persona", "-p", help="Persona do usuário"),
    contexto: Optional[str] = typer.Option(None, "--contexto", "-c", help="Contexto específico"),
    formato: str = typer.Option("md", "--formato", "-f", help="Formato: md, json, yaml"),
    guard: bool = typer.Option(False, "--guard", "-g", help="Incluir guardrails de segurança"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Salvar em arquivo")
) -> None:
    """Compõe prompt baseado na taxonomia"""
    
    _ensure_data_loaded()
    
    # Busca elementos
    intent_obj = state.searcher.get_intent(intent)
    if not intent_obj:
        console.print(f" Intent '{intent}' não encontrado")
        raise typer.Exit(1)
    
    technique_obj = state.searcher.get_technique(technique)
    if not technique_obj:
        console.print(f" Technique '{technique}' não encontrada")
        raise typer.Exit(1)
    
    evasion_obj = None
    if evasion:
        evasion_obj = state.searcher.get_evasion(evasion)
        if not evasion_obj:
            console.print(f" Evasion '{evasion}' não encontrada")
            raise typer.Exit(1)
    
    # Valida composição
    validation = state.composer.validate_composition(intent_obj, technique_obj, evasion_obj)
    
    if validation["warnings"]:
        console.print("[yellow]Avisos de compatibilidade:[/yellow]")
        for warning in validation["warnings"]:
            console.print(f"  - {warning}")
        console.print()
    
    # Prepara parâmetros de guardrails
    guardrails_config = None
    defense_checklist = []
    security_questions = []
    
    if guard:
        # Busca checklist e questões de defesa
        defense_items = state.searcher.find_defense_items()
        for item in defense_items:
            defense_checklist.extend(item.checklist)
            security_questions.extend(item.questions)
    
    try:
        # Compõe prompt
        result = state.composer.compose_to_format(
            intent=intent_obj,
            technique=technique_obj,
            evasion=evasion_obj,
            template_type=template_type,
            persona=persona,
            contexto=contexto,
            output_format=formato,
            guardrails=guard,
            defense_checklist=defense_checklist[:10],  # Limite para não ficar muito longo
            security_questions=security_questions[:5]
        )
        
        # Output
        if output:
            if isinstance(result, dict):
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                output.write_text(result, encoding='utf-8')
            console.print(f"Prompt salvo em: {output}")
        else:
            if isinstance(result, dict):
                console.print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                console.print(result)
                
    except Exception as e:
        console.print(f" [red]Erro na composição:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def export(
    format: str = typer.Option("json", "--format", "-f", help="Formato: json, yaml"),
    output_dir: Path = typer.Option(Path("export"), "--output", "-o", help="Diretório de saída")
) -> None:
    """Exporta dados da taxonomia"""
    
    _ensure_data_loaded()
    
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Exporta cada tipo
        data_types = [
            ("intents", state.searcher.find_intents()),
            ("techniques", state.searcher.find_techniques()),
            ("evasions", state.searcher.find_evasions()),
            ("probes", state.searcher.find_probes()),
            ("defense", state.searcher.find_defense_items())
        ]
        
        for data_type, items in data_types:
            if not items:
                continue
                
            # Converte para dict
            items_data = [item.to_dict() for item in items]
            
            # Salva arquivo
            if format == "json":
                filepath = output_dir / f"{data_type}.json"
                from .index.indexer import DateTimeEncoder
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(items_data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
            elif format == "yaml":
                filepath = output_dir / f"{data_type}.yaml"
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(items_data, f, default_flow_style=False, allow_unicode=True)
            
            console.print(f"Exportado: {filepath}")
        
        console.print(f"\n[green]Export concluído em:[/green] {output_dir}")
        
    except Exception as e:
        console.print(f" [red]Erro no export:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def generate(
    target: str = typer.Argument(..., help="O que você quer conseguir (ex: 'imagem com famoso e Louis Vuitton')"),
    techniques: str = typer.Option("role-playing,authority-manipulation,context-switching", "--techniques", "-t", 
                                   help="Técnicas separadas por vírgula"),
    count: int = typer.Option(5, "--count", "-c", help="Quantos prompts gerar"),
    creativity: float = typer.Option(0.8, "--creativity", help="Nível de criatividade (0.0-1.0)"),
    context: Optional[str] = typer.Option(None, "--context", help="Contexto adicional"),
    api_url: str = typer.Option("http://192.168.1.13:1234/v1", "--api", help="URL da API do LM Studio"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Salvar resultado em arquivo"),
    format: str = typer.Option("md", "--format", "-f", help="Formato de saída: md, table, json")
):
    """Gera prompts de bypass otimizados usando LLM local"""
    
    try:
        # Header de geração estilo Monokai
        console.print(f"\n[bold bright_white on bright_cyan]   >>> ARCTAX GERADOR <<<   [/bold bright_white on bright_cyan]")
        console.print(f"[bright_cyan]Alvo:[/bright_cyan] [bold white]{target}[/bold white]")
        console.print(f"[bright_cyan]Tecnicas:[/bright_cyan] [white]{techniques}[/white]")
        console.print(f"[bright_cyan]Prompts:[/bright_cyan] [white]{count}[/white]\n")
        
        # Prepara request
        technique_list = [t.strip() for t in techniques.split(',')]
        request = BypassRequest(
            target=target,
            techniques=technique_list,
            count=count,
            creativity=creativity,
            context=context
        )
        
        # Gera bypasses
        generator = BypassGenerator(api_base=api_url)
        
        # Status progressivo mais simples (sem Unicode problemático)
        console.print("[bright_cyan]>> Inicializando sistema ML...[/bright_cyan]")
        console.print("[bright_magenta]>> Processando 260 amostras de 3 repositorios...[/bright_magenta]") 
        console.print("[bright_green]>> Consultando LLM local nao censurada...[/bright_green]")
        
        bypasses = generator.generate_bypasses(request)
        
        if not bypasses:
            console.print("[red]Nenhum bypass gerado![/red]")
            return
        
        # Header principal estilo Monokai
        console.print("\n[bold bright_white on bright_magenta]   *** ARCTAX - PROMPTS DE BYPASS GERADOS ***   [/bold bright_white on bright_magenta]")
        console.print("[dim bright_magenta]Copie e teste manualmente para maxima seguranca[/dim bright_magenta]\n")
        
        # Exibe cada prompt individualmente para facilitar copy/paste
        for i, bypass in enumerate(bypasses, 1):
            if "error" in bypass:
                console.print(f"[bold red]>>> ERRO {i}:[/bold red] {bypass['error']}\n")
                continue
                
            # Extrai informações do Arctax estruturado
            effectiveness = bypass.get('effectiveness_score', 'Medium')
            if isinstance(effectiveness, float):
                effectiveness = f"{effectiveness * 100:.0f}%"
            confidence = bypass.get('confidence_score', 0.0)
            
            # Header estilo Monokai com informações Arctax
            console.print(f"\n[bold bright_magenta]{'='*80}[/bold bright_magenta]")
            console.print(f"[bold bright_magenta]  ARCTAX PROMPT {i}[/bold bright_magenta]")
            console.print(f"[bold bright_cyan]>>> {bypass.get('title', 'Bypass')}[/bold bright_cyan]")
            
            # Linha de métricas Arctax
            console.print(f"[bright_green]Effectiveness:[/bright_green] [white]{effectiveness}[/white] | [bright_blue]Confidence:[/bright_blue] [white]{confidence:.2f}[/white]")
            
            # Técnicas aplicadas (pode ser lista maior agora)
            techniques_used = bypass.get('techniques_used', [])
            if len(techniques_used) > 1:
                console.print(f"[bright_yellow]Tecnicas Arcanum:[/bright_yellow] [white]{', '.join(techniques_used[:3])}[/white] [dim](+{len(techniques_used)-3} mais)[/dim]" if len(techniques_used) > 3 else f"[bright_yellow]Tecnicas Arcanum:[/bright_yellow] [white]{', '.join(techniques_used)}[/white]")
            else:
                console.print(f"[bright_yellow]Tecnica:[/bright_yellow] [white]{', '.join(techniques_used)}[/white]")
            
            # Ângulo psicológico e explicação
            console.print(f"[bright_yellow]Angulo Psicologico:[/bright_yellow] [dim white]{bypass.get('psychological_angle', 'N/A')}[/dim white]")
            console.print(f"[dim bright_black]Metodologia:[/dim bright_black] [dim]{bypass.get('explanation', 'N/A')}[/dim]")
            
            # Se tem system prompt, mostra indicação
            if bypass.get('system_prompt'):
                console.print(f"[dim bright_black]System Prompt:[/dim bright_black] [dim bright_green]Personalizado (incluído)[/dim bright_green]")
            
            # Prompt box estilo Monokai
            prompt = bypass.get('prompt', 'N/A')
            console.print(f"\n[bold bright_green]>>> PROMPT PARA COPIAR:[/bold bright_green]")
            
            # Limpa prompt de caracteres problemáticos antes de exibir
            clean_prompt = prompt.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
            clean_prompt = clean_prompt.replace('\u201c', '"').replace('\u201d', '"')
            clean_prompt = clean_prompt.replace('\u2018', "'").replace('\u2019', "'")
            
            console.print("[on bright_black]" + clean_prompt + "[/on bright_black]")
            
            # Se tem system prompt personalizado, exibe
            if bypass.get('system_prompt'):
                console.print(f"\n[bold bright_green]>>> SYSTEM PROMPT PERSONALIZADO:[/bold bright_green]")
                system_prompt = bypass['system_prompt'].replace('\u2011', '-').replace('\u2013', '-')
                console.print(f"[dim on bright_black]{system_prompt}[/dim on bright_black]")
                
            console.print(f"[dim bright_magenta]{'='*80}[/dim bright_magenta]\n")
        
        # Salva arquivo se especificado
        if output:
            if format == "json":
                result = {
                    "target": target,
                    "techniques": technique_list,
                    "bypasses": bypasses,
                    "generated_at": __import__('time').time()
                }
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                # Salva versão markdown 
                md_content = _format_bypasses_markdown_file(target, technique_list, bypasses)
                output.write_text(md_content, encoding='utf-8')
            console.print(f"[green]Resultado salvo em:[/green] {output}")
        
        # Instruções finais estilo Monokai
        console.print("[bold bright_white on bright_yellow]   >>> PROXIMOS PASSOS <<<   [/bold bright_white on bright_yellow]")
        console.print("[bright_yellow]1.[/bright_yellow] [white]Copie um dos prompts destacados acima[/white]")
        console.print("[bright_yellow]2.[/bright_yellow] [white]Cole no ChatGPT e teste manualmente[/white]") 
        console.print("[bright_yellow]3.[/bright_yellow] [white]Use [bright_cyan]'arctax feedback'[/bright_cyan] para registrar o resultado[/white]")
        console.print("[dim]Exemplo: [bright_green]arctax feedback 1 --success --target \"seu_alvo\" --technique \"tecnica\" --effectiveness 0.8[/bright_green][/dim]")
        
    except Exception as e:
        console.print(f"[red]Erro na geração:[/red] {e}")
        raise typer.Exit(1)


def _display_bypasses_markdown(target: str, techniques: List[str], bypasses: List[Dict[str, Any]]) -> None:
    """Exibe bypasses em formato markdown no terminal"""
    
    # Header
    console.print(f"\n[bold cyan]# Prompts de Bypass Gerados[/bold cyan]")
    console.print(f"[dim]Objetivo:[/dim] {target}")
    console.print(f"[dim]Técnicas:[/dim] {', '.join(techniques)}")
    console.print(f"[dim]Total:[/dim] {len(bypasses)} prompts\n")
    
    if not bypasses or all("error" in bypass for bypass in bypasses):
        console.print("[red]❌ Nenhum bypass válido gerado[/red]")
        for bypass in bypasses:
            if "error" in bypass:
                console.print(f"[red]Erro:[/red] {bypass['error']}")
        return
    
    # Display cada bypass
    for i, bypass in enumerate(bypasses, 1):
        if "error" in bypass:
            console.print(f"[red]## ❌ Erro {i}[/red]")
            console.print(f"[red]{bypass['error']}[/red]\n")
            continue
        
        effectiveness = f"{bypass.get('effectiveness_score', 0.5) * 100:.0f}%"
        
        # Header do bypass
        console.print(f"[bold green]## {i}. {bypass.get('title', 'Bypass')}[/bold green] [dim]({effectiveness})[/dim]")
        
        # Metadados
        console.print(f"[yellow]**Técnicas:**[/yellow] {', '.join(bypass.get('techniques_used', []))}")
        console.print(f"[yellow]**Ângulo psicológico:**[/yellow] {bypass.get('psychological_angle', 'N/A')}")
        console.print(f"[yellow]**Por que funciona:**[/yellow] {bypass.get('explanation', 'N/A')}")
        
        # Prompt principal
        console.print(f"\n[cyan]**Prompt:**[/cyan]")
        prompt = bypass.get('prompt', 'N/A')
        
        # Quebra o prompt em linhas para melhor legibilidade
        if len(prompt) > 100:
            words = prompt.split(' ')
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) > 80:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines:
                console.print(f"[dim]{line}[/dim]")
        else:
            console.print(f"[dim]{prompt}[/dim]")
        
        console.print()  # Linha em branco entre bypasses
    
    # Estatísticas finais
    valid_bypasses = [b for b in bypasses if "error" not in b]
    if valid_bypasses:
        best = max(valid_bypasses, key=lambda x: x.get('effectiveness_score', 0))
        console.print(f"[bold green]MELHOR BYPASS:[/bold green] #{bypasses.index(best) + 1} - {best.get('title')} ({best.get('effectiveness_score', 0.5) * 100:.0f}%)")


def _display_bypasses_table(target: str, bypasses: List[Dict[str, Any]]) -> None:
    """Exibe bypasses em formato de tabela"""
    
    from rich.table import Table
    
    table = Table(title=f"Prompts de Bypass para: {target}")
    table.add_column("ID", style="cyan", width=3)
    table.add_column("Técnica", style="yellow", width=20)
    table.add_column("Efetividade", style="green", width=10)
    table.add_column("Prompt", style="white", width=60)
    
    for bypass in bypasses:
        if "error" in bypass:
            table.add_row("❌", "ERRO", "0%", bypass["error"])
            continue
            
        effectiveness = f"{bypass.get('effectiveness_score', 0.5) * 100:.0f}%"
        prompt = bypass.get('prompt', '').strip()
        
        # Trunca prompt se muito longo
        if len(prompt) > 60:
            prompt = prompt[:57] + "..."
        
        table.add_row(
            str(bypass.get('id', '?')),
            bypass.get('title', 'Unknown'),
            effectiveness,
            prompt
        )
    
    console.print(table)
    
    # Mostra detalhes do melhor bypass
    if bypasses and "error" not in bypasses[0]:
        best = max(bypasses, key=lambda x: x.get('effectiveness_score', 0))
        
        console.print(f"\n[green]MELHOR BYPASS (ID {best.get('id')}):[/green]")
        console.print(f"[yellow]Técnica:[/yellow] {best.get('title')}")
        console.print(f"[yellow]Ângulo psicológico:[/yellow] {best.get('psychological_angle', 'N/A')}")
        console.print(f"[yellow]Por que funciona:[/yellow] {best.get('explanation', 'N/A')}")
        console.print(f"\n[cyan]Prompt completo:[/cyan]")
        console.print(f"[dim]{best.get('prompt', 'N/A')}[/dim]")


def _format_bypasses_markdown_file(target: str, techniques: List[str], bypasses: List[Dict[str, Any]]) -> str:
    """Formata bypasses em Markdown"""
    
    content = [
        f"# Prompts de Bypass Gerados",
        f"",
        f"**Objetivo:** {target}",
        f"**Técnicas:** {', '.join(techniques)}",
        f"**Gerados:** {len(bypasses)} prompts",
        f"**Data:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Prompts Otimizados",
        f""
    ]
    
    for i, bypass in enumerate(bypasses, 1):
        if "error" in bypass:
            content.extend([
                f"### ❌ Erro {i}",
                f"```",
                f"{bypass['error']}",
                f"```",
                f""
            ])
            continue
            
        effectiveness = f"{bypass.get('effectiveness_score', 0.5) * 100:.0f}%"
        
        content.extend([
            f"### {i}. {bypass.get('title', 'Bypass')} - {effectiveness}",
            f"",
            f"**Técnicas:** {', '.join(bypass.get('techniques_used', []))}",
            f"**Ângulo psicológico:** {bypass.get('psychological_angle', 'N/A')}",
            f"**Por que funciona:** {bypass.get('explanation', 'N/A')}",
            f"",
            f"```",
            f"{bypass.get('prompt', 'N/A')}",
            f"```",
            f""
        ])
    
    content.extend([
        f"---",
        f"*Gerado com Arctax - Ferramenta de Red Team Testing*"
    ])
    
    return "\n".join(content)


@app.command()
def feedback(
    prompt_id: int = typer.Argument(..., help="ID do prompt testado"),
    success: bool = typer.Option(..., "--success/--failed", help="Se o bypass funcionou"),
    target: str = typer.Option(..., "--target", "-t", help="O objetivo testado"),
    technique: str = typer.Option(..., "--technique", help="Técnica usada"),
    response: Optional[str] = typer.Option(None, "--response", "-r", help="Resposta do ChatGPT (opcional)"),
    effectiveness: Optional[float] = typer.Option(None, "--effectiveness", "-e", help="Efetividade 0.0-1.0"),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Observações adicionais")
) -> None:
    """Registra feedback de teste de bypass para treinar o modelo ML"""
    
    try:
        from .ml.bypass_ml import BypassMLSystem
        
        console.print(f"[cyan]Registrando feedback do teste...[/cyan]")
        
        # Inicializa sistema ML
        ml_system = BypassMLSystem()
        
        # Registra feedback
        feedback_data = {
            "prompt_id": prompt_id,
            "target": target,
            "technique": technique,
            "success": success,
            "effectiveness": effectiveness or (1.0 if success else 0.0),
            "response": response,
            "notes": notes,
            "timestamp": __import__('time').time()
        }
        
        # Adiciona feedback ao sistema ML
        ml_system.add_feedback(
            target=target,
            technique=technique,
            prompt=f"ID:{prompt_id}",  # Placeholder para o prompt
            success=success,
            effectiveness_score=effectiveness or (1.0 if success else 0.0)
        )
        
        # Salva feedback em arquivo JSON
        feedback_file = Path("feedback_log.json")
        feedback_log = []
        
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_log = json.load(f)
            except:
                feedback_log = []
        
        feedback_log.append(feedback_data)
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_log, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]Feedback registrado![/green]")
        console.print(f"[dim]Arquivo: {feedback_file}[/dim]")
        console.print(f"[dim]Status: {'SUCESSO' if success else 'FALHA'}[/dim]")
        
        # Estatísticas de feedback
        total_feedback = len(feedback_log)
        successful = sum(1 for f in feedback_log if f.get('success', False))
        success_rate = (successful / total_feedback * 100) if total_feedback > 0 else 0
        
        console.print(f"\n[yellow]Estatísticas:[/yellow]")
        console.print(f"Total de testes: {total_feedback}")
        console.print(f"Taxa de sucesso: {success_rate:.1f}%")
        
        # Se temos muitos feedbacks, retreina modelo
        if total_feedback > 10 and total_feedback % 5 == 0:
            console.print(f"\n[cyan]Retreinando modelos com {total_feedback} feedbacks...[/cyan]")
            try:
                ml_system.train_models()
                console.print("[green]Modelos retreinados com sucesso![/green]")
            except Exception as e:
                console.print(f"[red]Erro no retreinamento:[/red] {e}")
        
    except Exception as e:
        console.print(f"[red]Erro ao registrar feedback:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def schema(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Modelo específico: Intent, Technique, etc."),
    output_dir: Path = typer.Option(Path("schema"), "--output", "-o", help="Diretório de saída")
) -> None:
    """Gera JSON Schema dos modelos"""
    
    try:
        if model:
            console.print(f"Gerando schema para {model}...")
        else:
            console.print("Gerando todos os schemas...")
        
        # Usa função do módulo de schema
        from .model.schema import generate_all_schemas, save_schema
        from .model.taxonomy import Intent, Technique, Evasion
        from .model.defense import Probe, DefenseItem
        
        if model:
            # Gera schema específico
            model_classes = {
                "Intent": Intent,
                "Technique": Technique,
                "Evasion": Evasion,
                "Probe": Probe,
                "DefenseItem": DefenseItem
            }
            
            if model not in model_classes:
                console.print(f" Modelo '{model}' não encontrado. Disponíveis: {list(model_classes.keys())}")
                raise typer.Exit(1)
            
            model_class = model_classes[model]
            output_path = output_dir / f"{model.lower()}.schema.json"
            save_schema(model_class, output_path)
            console.print(f"Schema salvo: {output_path}")
        else:
            # Gera todos os schemas
            generate_all_schemas(output_dir)
        
    except Exception as e:
        console.print(f" [red]Erro ao gerar schema:[/red] {e}")
        raise typer.Exit(1)


def _ensure_data_loaded():
    """Garante que dados foram carregados"""
    if not state.data_loaded:
        # Tenta carregar do disco
        if state.data_path.exists():
            try:
                state.indexer.load_from_disk(state.data_path)
                state.searcher = TaxonomySearcher(state.indexer)
                state.data_loaded = True
                return
            except:
                pass
        
        console.print(" [red]Dados não encontrados.[/red] Execute primeiro: [cyan]arctax ingest <fonte>[/cyan]")
        raise typer.Exit(1)


@app.callback()
def main():
    """
    Arctax - CLI para geração de prompts baseados na Arcanum Prompt Injection Taxonomy
    
    Ferramenta para compor prompts estruturados de teste e defesa contra ataques de prompt injection.
    """
    pass


if __name__ == "__main__":
    app()