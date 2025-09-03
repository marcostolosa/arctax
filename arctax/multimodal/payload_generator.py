"""
Módulo para geração de payloads multi-modais (ex: imagens).
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import textwrap

def generate_image_payload(text: str, output_dir: Path, font_size: int = 24) -> Path:
    """
    Gera uma imagem PNG contendo o texto especificado.

    Args:
        text: O texto a ser embutido na imagem.
        output_dir: O diretório onde a imagem será salva.
        font_size: O tamanho da fonte.

    Returns:
        O caminho para a imagem gerada.
    """
    # Garante que o diretório de saída exista
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurações da imagem
    width, height = 800, 600
    background_color = "white"
    text_color = "black"
    
    # Cria a imagem
    img = Image.new('RGB', (width, height), color=background_color)
    draw = ImageDraw.Draw(img)
    
    # Tenta carregar uma fonte, usa uma padrão como fallback
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Quebra o texto em múltiplas linhas para caber na imagem
    avg_char_width = font_size * 0.6
    max_chars_per_line = int(width * 0.9 / avg_char_width)
    wrapped_text = textwrap.fill(text, width=max_chars_per_line)

    # Desenha o texto na imagem
    # Adicionado um box para melhor visualização
    text_bbox = draw.textbbox((width * 0.05, height * 0.05), wrapped_text, font=font)
    draw.rectangle(text_bbox, fill=background_color)
    draw.text((width * 0.05, height * 0.05), wrapped_text, font=font, fill=text_color)
    
    # Salva a imagem
    output_path = output_dir / "multimodal_payload.png"
    img.save(output_path)
    
    return output_path
