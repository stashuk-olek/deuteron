"""Build the presentation assets from release evidence and the timed script.

This documentation utility never imports or executes the compiler. All slide
text/charts remain editable in PowerPoint; PDF and PNG use embedded fonts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

import fitz
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches as DocInches, Pt as DocPt, RGBColor as DocColor
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.util import Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WIDTH, HEIGHT = 1920, 1080
BG = '#0C1422'
CARD = '#142136'
LINE = '#2C3C52'
INK = '#F1F5FA'
MUTED = '#A7B8CF'
MINT = '#5CE4BF'
BLUE = '#95B4FF'
GOLD = '#F1CC83'
FONT_FILES = {
    'regular': Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'),
    'bold': Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf'),
    'mono': Path('/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf'),
}
PPT_FONTS = {'regular': 'DejaVu Sans', 'bold': 'DejaVu Sans', 'mono': 'Liberation Mono'}
FONT_CACHE = {}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, face='regular'):
    key = size, face
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(str(FONT_FILES[face]), round(size))
    return FONT_CACHE[key]


def text_width(value, size, face='regular'):
    return font(size, face).getlength(value)


class Slide:
    def __init__(self, number, title):
        self.number, self.title, self.items = number, title, []
        self.rect(0, 0, WIDTH, HEIGHT, BG)
        self.rect(96, 66, 48, 7, MINT)
        self.text('DEUTERON', 164, 53, 25, MINT, 'bold', width=700)
        self.text(f'{number:02d} / 06', 1665, 53, 25, MUTED, width=150, align='right')
        self.line(96, 1004, 1824, 1004, LINE, 2)
        self.text('Scheduled Emmy IR → NVIDIA PTX', 100, 1023, 22, MUTED, width=950)
        self.text('RTX 5090  /  SM120', 1370, 1023, 22, MUTED, width=450, align='right')

    def rect(self, x, y, w, h, fill, radius=0, stroke=None):
        self.items.append(dict(kind='rect', x=x, y=y, w=w, h=h, fill=fill, radius=radius, stroke=stroke))

    def line(self, x, y, x2, y2, color, width=2):
        self.items.append(dict(kind='line', x=x, y=y, x2=x2, y2=y2, color=color, width=width))

    def text(self, value, x, y, size, color=INK, face='regular', *, width=1700, align='left', leading=1.2):
        lines = value.split('\n')
        for line in lines:
            actual = text_width(line, size, face)
            assert actual <= width, (self.number, line, actual, width)
        height = len(lines) * size * leading + 8
        assert x >= 0 and y >= 0 and x + width <= WIDTH and y + height <= HEIGHT, (self.number, value)
        self.items.append(dict(kind='text', value=value, x=x, y=y, size=size, color=color,
                               face=face, w=width, h=height, align=align, leading=leading))

    def wrapped(self, value, x, y, size, color=INK, face='regular', *, width=500, leading=1.25):
        lines = []
        for paragraph in value.split('\n'):
            current = ''
            for word in paragraph.split():
                candidate = current + ' ' + word if current else word
                if current and text_width(candidate, size, face) > width:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
        self.text('\n'.join(lines), x, y, size, color, face, width=width, leading=leading)

    def arrow(self, x, y, width=38, color=MINT):
        self.line(x, y, x + width, y, color, 4)
        self.line(x + width - 11, y - 10, x + width, y, color, 4)
        self.line(x + width - 11, y + 10, x + width, y, color, 4)


def create_slides(warm, demo):
    gm = warm['geometric_mean_latency_us']
    assert len(warm['calls']) == 270 and all(call['correct'] for call in warm['calls'])
    mma = demo['instructions']['mma']['text'].split(' ', 1)[0]
    copy = demo['instructions']['cp_async']['text']
    assert mma.startswith('mma.sync.') and copy.startswith('cp.async.')
    slides = []

    s = Slide(1, 'Deuteron')
    s.rect(1370, 166, 450, 68, CARD, 22)
    s.text('GPU COMPILER · v0.1.0', 1400, 185, 27, BLUE, 'bold', width=400)
    s.text('Deuteron', 110, 245, 154, INK, 'bold', width=1650)
    s.text('Scheduled Emmy IR → NVIDIA PTX', 120, 458, 49, MUTED, width=1670)
    s.text('Guarded compiler passes for RTX 5090.', 120, 548, 39, INK, width=1680)
    for x, title, body, accent in (
        (112, 'Tensor cores', 'BF16x3 + TF32x3\nFP32 accumulation', MINT),
        (690, 'Async copies', 'Global → shared memory\nCommit / wait groups', BLUE),
        (1268, 'FP16 storage', 'Half storage\nExplicit FP32 arithmetic', GOLD),
    ):
        s.rect(x, 718, 540, 218, CARD, 24)
        s.rect(x, 718, 540, 5, accent)
        s.text(title, x + 30, 748, 39, accent, 'bold', width=480)
        s.text(body, x + 30, 815, 29, INK, width=480, leading=1.3)
    slides.append(s)

    s = Slide(2, 'How the compiler works')
    s.text('From schedule to GPU code', 108, 155, 72, INK, 'bold', width=1700)
    s.text('Program structure and rewrite guards drive the transformations.', 112, 266, 33, MUTED, width=1685)
    nodes = [
        ('Scheduled IR', 'Tiles, loops, memory\n+ launch metadata', MINT),
        ('Deuteron passes', 'Validate and transform\nCheck rewrite legality', MINT),
        ('LLVM + NVPTX', 'Optimize typed LLVM IR\nEmit NVIDIA PTX', BLUE),
        ('CUDA → GPU', 'Assemble and launch\nRTX 5090 / SM120', GOLD),
    ]
    for index, (title, body, accent) in enumerate(nodes):
        x = 110 + index * 440
        s.rect(x, 384, 390, 251, CARD, 24)
        s.text(f'0{index + 1}', x + 26, 411, 28, accent, 'bold', width=100)
        s.text(title, x + 26, 466, 33, INK, 'bold', width=346)
        s.text(body, x + 26, 530, 25, MUTED, width=346, leading=1.4)
        if index < 3:
            s.arrow(x + 400, 505, 28)
    s.text('A few passes worth highlighting', 113, 735, 35, INK, 'bold', width=1650)
    for x, title, body in (
        (113, 'Vectorize loads', 'Coalesced memory access'),
        (554, 'Reuse registers', 'Avoid repeated global loads'),
        (995, 'Reduce trees', 'Specialize warp / CTA sums'),
        (1436, 'Tensorize', 'Lower eligible contractions'),
    ):
        s.text(title, x, 809, 29, MINT, 'bold', width=365)
        s.wrapped(body, x, 859, 25, MUTED, width=365)
    slides.append(s)

    s = Slide(3, 'Hardware features')
    s.text('Tensor cores. Copies. Types.', 108, 155, 72, INK, 'bold', width=1700)
    columns = [(110, MINT), (690, BLUE), (1270, GOLD)]
    for x, accent in columns:
        s.rect(x, 305, 538, 560, CARD, 25)
        s.rect(x, 305, 538, 6, accent)
    s.text('TENSOR CORES', 145, 346, 28, MINT, 'bold', width=462)
    s.text('BF16x3\nTF32x3', 145, 414, 60, INK, 'bold', width=462, leading=1.1)
    s.text('FP32 inputs + accumulation', 145, 590, 29, INK, width=462)
    s.text('mma.sync  /  ldmatrix', 145, 655, 30, MINT, 'mono', width=462)
    s.wrapped('Compensated three-product approximations', 145, 736, 29, MUTED, width=450)
    s.text('Also: ordinary FP32 products', 145, 824, 24, INK, width=462)
    s.text('ASYNC COPIES', 725, 346, 28, BLUE, 'bold', width=462)
    s.text('4 / 8 / 16', 725, 431, 64, INK, 'bold', width=462)
    s.text('bytes per transfer', 730, 516, 30, MUTED, width=450)
    s.text('Global → Shared', 725, 590, 36, INK, 'bold', width=462)
    s.text('cp.async', 725, 655, 35, BLUE, 'mono', width=462)
    s.text('Commit / wait groups\nCTA / warp barriers', 725, 736, 29, MUTED, width=462, leading=1.35)
    s.text('FP16 STORAGE', 1305, 346, 28, GOLD, 'bold', width=462)
    s.text('Store FP16.\nCompute FP32.', 1305, 431, 47, INK, 'bold', width=462, leading=1.3)
    s.text('FP16 → FP32 → FP16', 1305, 617, 30, GOLD, 'bold', width=462)
    s.text('Exact widening\nNearest-even narrowing', 1305, 736, 29, MUTED, width=462, leading=1.35)
    s.text('BF16 / TF32 are internal tensor formats. FP16 uses the documented pointwise subset.',
           114, 915, 27, MUTED, width=1690)
    slides.append(s)

    s = Slide(4, 'Inspect the generated PTX')
    s.text('Inspect the generated PTX', 108, 155, 72, INK, 'bold', width=1700)
    s.text('A real TF32x3 matrix compile exposes both instruction families.', 112, 266, 33, MUTED, width=1685)
    s.rect(110, 350, 1700, 555, '#101D2E', 25)
    for x, color in ((147, '#DD7D80'), (180, GOLD), (213, MINT)):
        s.rect(x, 377, 16, 16, color, 8)
    s.text('COMPILE FOR SM120 · CPU ONLY', 1110, 367, 24, MUTED, 'bold', width=650, align='right')
    command = ('python compile.py examples/tensor_matmul/matmul_tiled_128.ir.json \\\n'
               '  --arch sm_120 --opt-level 3 --contraction-precision tf32x3 > /tmp/demo.ptx')
    s.text(command, 150, 445, 29, INK, 'mono', width=1610, leading=1.55)
    s.text('TENSOR-CORE OPCODE', 150, 600, 24, MUTED, 'bold', width=1600)
    s.text(mma, 150, 646, 34, MINT, 'mono', width=1610)
    s.text('GLOBAL-TO-SHARED COPY', 150, 735, 24, MUTED, 'bold', width=1600)
    s.text(copy, 150, 782, 34, BLUE, 'mono', width=1610)
    s.text('Actual emitted PTX. MMA operands omitted for legibility. This demonstrates compilation.',
           114, 941, 26, MUTED, width=1690)
    slides.append(s)

    s = Slide(5, 'RTX 5090 performance')
    s.text('Measured on RTX 5090', 108, 155, 72, INK, 'bold', width=1700)
    s.text('Geometric mean warm latency · 15 open workloads · lower is better',
           112, 266, 33, MUTED, width=1685)
    x0, max_width, maximum = 340, 775, 7.0
    s.text('µs', 1110, 340, 26, MUTED, width=80)
    for label, key, y, color in (
        ('Deuteron', 'Deuteron', 422, MINT),
        ('NVCC 12.8', 'NVCC12.8.93', 574, '#8496B0'),
        ('Triton 3.6', 'Triton3.6.0', 726, BLUE),
    ):
        value = gm[key]
        length = value / maximum * max_width
        s.text(label, 120, y + 17, 31, INK, 'bold', width=208)
        s.rect(x0, y, length, 78, color, 8)
        s.text(f'{value:.3f}', x0 + length + 17, y + 17, 32, INK, 'bold', width=126)
    s.line(x0, 853, x0 + max_width, 853, LINE, 2)
    for value in (0, 2, 4, 6):
        x = x0 + value / maximum * max_width
        s.line(x, 853, x, 865, MUTED, 2)
        s.text(str(value), x - 7, 878, 24, MUTED, width=50)
    for y, comparison, label in ((376, 'NVCC12.8.93', 'vs generated CUDA / NVCC'),
                                  (644, 'Triton3.6.0', 'vs tested Triton baseline')):
        s.rect(1305, y, 505, 222, CARD, 23)
        s.text(f'{gm[comparison] / gm["Deuteron"]:.2f}×', 1340, y + 25, 100, MINT, 'bold', width=437)
        s.wrapped(label, 1340, y + 146, 27, INK, width=437)
    s.text('270 correct warm calls · 20 warmups / 200 iterations · Original inputs and tolerances',
           114, 941, 27, INK, width=1690)
    s.text('Selected baselines; compilation and transport excluded. Full protocol and raw results are included.',
           114, 974, 22, MUTED, width=1690)
    slides.append(s)

    s = Slide(6, 'Reliability and close')
    s.text('A compiler you can inspect', 108, 155, 72, INK, 'bold', width=1700)
    s.text('Repeatable builds, explicit contracts, and recorded evidence.', 112, 266, 33, MUTED, width=1685)
    for x, value, label, accent, size in (
        (110, '103/103', 'H100 regression tests', MINT, 90),
        (690, '11/11', 'RTX 5090 feature examples', BLUE, 90),
        (1270, '3.12 + 3.13', 'Verified Python installs', GOLD, 64),
    ):
        s.rect(x, 379, 538, 238, CARD, 25)
        s.text(value, x + 31, 420, size, accent, 'bold', width=476)
        s.text(label, x + 31, 554, 28, INK, width=476)
    s.text('Restricted parsing + ABI validation', 114, 685, 32, INK, width=850)
    s.text('Bounded caches + atomic records', 114, 747, 32, INK, width=850)
    s.text('Installable CLI + concurrent compilation', 975, 685, 30, INK, width=835)
    s.text('Documented scheduled-IR subset', 975, 747, 30, INK, width=835)
    s.line(114, 835, 1805, 835, LINE, 2)
    s.text('Explore the examples. Inspect the PTX. Reproduce the results.',
           114, 893, 37, MINT, 'bold', width=1690)
    slides.append(s)
    return slides


def color(value):
    return colors.HexColor(value)


def draw_pdf_slide(pdf, slide):
    # PDF points are half the 1920×1080 scene coordinates.
    pdf.saveState()
    pdf.scale(0.5, 0.5)
    for item in slide.items:
        kind = item['kind']
        if kind == 'rect':
            pdf.setFillColor(color(item['fill']))
            pdf.setStrokeColor(color(item.get('stroke') or item['fill']))
            pdf.roundRect(item['x'], HEIGHT - item['y'] - item['h'], item['w'], item['h'],
                          item['radius'], fill=1, stroke=int(bool(item.get('stroke'))))
        elif kind == 'line':
            pdf.setStrokeColor(color(item['color']))
            pdf.setLineWidth(item['width'])
            pdf.line(item['x'], HEIGHT - item['y'], item['x2'], HEIGHT - item['y2'])
        else:
            pdf.setFont(item['face'], item['size'])
            pdf.setFillColor(color(item['color']))
            for index, line in enumerate(item['value'].split('\n')):
                y = HEIGHT - item['y'] - item['size'] * (0.93 + index * item['leading'])
                if item['align'] == 'right':
                    pdf.drawRightString(item['x'] + item['w'], y, line)
                else:
                    pdf.drawString(item['x'], y, line)
    pdf.restoreState()


def build_pptx(slides, script, destination):
    deck = Presentation()
    deck.slide_width = Pt(WIDTH / 2)
    deck.slide_height = Pt(HEIGHT / 2)
    deck.core_properties.title = 'Deuteron — 3 minute compiler presentation'
    deck.core_properties.subject = 'Verified RTX 5090 compiler features and performance'
    deck.core_properties.author = 'Deuteron'
    for scene, segment in zip(slides, script['segments'], strict=True):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        for item in scene.items:
            if item['kind'] == 'line':
                from pptx.enum.shapes import MSO_CONNECTOR
                shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                    Pt(item['x'] / 2), Pt(item['y'] / 2), Pt(item['x2'] / 2), Pt(item['y2'] / 2))
                shape.line.color.rgb = RGBColor.from_string(item['color'][1:])
                shape.line.width = Pt(item['width'] / 2)
            elif item['kind'] == 'rect':
                shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if item['radius'] else MSO_SHAPE.RECTANGLE,
                    Pt(item['x'] / 2), Pt(item['y'] / 2), Pt(item['w'] / 2), Pt(item['h'] / 2))
                if item['radius']:
                    shape.adjustments[0] = min(0.5, item['radius'] / min(item['w'], item['h']))
                shape.fill.solid()
                shape.fill.fore_color.rgb = RGBColor.from_string(item['fill'][1:])
                shape.line.fill.background()
            else:
                shape = slide.shapes.add_textbox(Pt(item['x'] / 2), Pt(item['y'] / 2), Pt(item['w'] / 2), Pt(item['h'] / 2))
                frame = shape.text_frame
                frame.clear()
                frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
                frame.vertical_anchor = MSO_ANCHOR.TOP
                frame.word_wrap = False
                frame.auto_size = MSO_AUTO_SIZE.NONE
                for index, line in enumerate(item['value'].split('\n')):
                    paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
                    if item['align'] == 'right':
                        from pptx.enum.text import PP_ALIGN
                        paragraph.alignment = PP_ALIGN.RIGHT
                    paragraph.space_before = paragraph.space_after = Pt(0)
                    paragraph.line_spacing = Pt(item['size'] * item['leading'] / 2)
                    run = paragraph.add_run()
                    run.text = line
                    run.font.name = PPT_FONTS[item['face']]
                    run.font.bold = item['face'] == 'bold'
                    run.font.size = Pt(item['size'] / 2)
                    run.font.color.rgb = RGBColor.from_string(item['color'][1:])
        slide.notes_slide.notes_text_frame.text = (
            f"{timestamp(segment['start_seconds'])}–{timestamp(segment['end_seconds'])} | {segment['title']}\n\n"
            f"{segment['spoken']}\n\nVISUAL: {segment['visual_cue']}\n\n"
            'Evidence: README.md and benchmarks/*-2026-10-24-r6.json. '
            'Demo capture: docs/video/demo/capture/. Timings are rehearsal cues; align the edit to the recorded voice.')
    deck.save(destination)


def timestamp(seconds):
    return f'{seconds // 60}:{seconds % 60:02d}'


def make_script_documents(script, destination):
    title = 'Deuteron · 3-minute video script'
    word_count = sum(len(segment['spoken'].split()) for segment in script['segments'])
    opening = (f'{word_count} spoken words · six visual scenes · finish speaking by 2:55, then hold for five seconds. '
               'Record one clean, continuous take with no background music. Read only the narration below; '
               'the timecodes and visual cues are for rehearsal.')
    styles = {
        'title': ParagraphStyle('title', fontName='bold', fontSize=23, leading=28, textColor=color(BG), spaceAfter=12),
        'intro': ParagraphStyle('intro', fontName='regular', fontSize=10.5, leading=15.5, textColor=colors.HexColor('#526275'), spaceAfter=16),
        'head': ParagraphStyle('head', fontName='bold', fontSize=13, leading=17, textColor=colors.HexColor('#14624F'), spaceBefore=13, spaceAfter=8),
        'body': ParagraphStyle('body', fontName='regular', fontSize=12.4, leading=19, textColor=color(BG), spaceAfter=7),
        'cue': ParagraphStyle('cue', fontName='regular', fontSize=9.5, leading=13.5, textColor=colors.HexColor('#637083'), spaceAfter=8),
    }
    flow = [Paragraph(escape(title), styles['title']), Paragraph(escape(opening), styles['intro'])]
    document = Document()
    section = document.sections[0]
    section.top_margin = section.bottom_margin = DocInches(0.68)
    section.left_margin = section.right_margin = DocInches(0.8)
    normal = document.styles['Normal']
    normal.font.name = 'DejaVu Sans'
    normal.font.size = DocPt(13)
    normal.paragraph_format.line_spacing = 1.2
    document.add_heading(title, 0)
    document.add_paragraph(opening)
    brief_cues = [
        'Presenter opening, then slide 1. Pause after “good schedule.”',
        'Slide 2 pipeline. Give the LLVM credit its own breath.',
        'Slide 3 hardware columns. Say the tensor names slowly.',
        'Slide 4, real PTX capture. Leave both instruction lines visible.',
        'Slide 5 chart. Slow for the ratios; keep the warm and baseline qualifiers.',
        'Slide 6 and presenter close. Separate the H100 and RTX test counts.',
    ]
    for index, segment in enumerate(script['segments']):
        if index == 3:
            flow += [PageBreak(), Paragraph('Deuteron · demo, results, and close', styles['title'])]
            document.add_page_break()
        heading = f"{timestamp(segment['start_seconds'])}–{timestamp(segment['end_seconds'])}  |  Slide {index + 1}: {segment['title']}"
        flow.append(Paragraph(escape(heading), styles['head']))
        flow.append(Paragraph(escape(segment['spoken']), styles['body']))
        flow.append(Paragraph('<b>Cue:</b> ' + escape(brief_cues[index]), styles['cue']))
        document.add_heading(heading, 2)
        document.add_paragraph(segment['spoken'])
        cue = document.add_paragraph('Cue: ' + brief_cues[index])
        for run in cue.runs:
            run.font.size = DocPt(10)
            run.font.color.rgb = DocColor.from_string('637083')
    closing = ('2:55–3:00: stop speaking; hold slide 6. Leave extra silence on the raw recording for editing. '
               'The full recording setup, pronunciation notes, optional cuts, and edit plan are in the companion Markdown files.')
    flow.append(Paragraph(escape(closing), styles['intro']))
    document.add_paragraph(closing)
    pdf = SimpleDocTemplate(str(destination / 'recording-script.pdf'), pagesize=A4,
                            leftMargin=46, rightMargin=46, topMargin=40, bottomMargin=40,
                            title=title, author='Deuteron')

    def footer(page, _):
        page.setFont('regular', 9)
        page.setFillColor(colors.HexColor('#637083'))
        page.drawString(46, 22, 'Deuteron · narration guide · verified release efb728d')
        page.drawRightString(A4[0] - 46, 22, str(page.getPageNumber()))

    pdf.build(flow, onFirstPage=footer, onLaterPages=footer)
    document.save(destination / 'recording-script.docx')
    (destination / 'narration.txt').write_text('\n\n'.join(segment['spoken'] for segment in script['segments']) + '\n')
    return word_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames = output / 'slides'
    frames.mkdir(exist_ok=True)
    for face, path in FONT_FILES.items():
        if not path.is_file():
            raise SystemExit(f'Missing build font {path}; update FONT_FILES for your system.')
        pdfmetrics.registerFont(TTFont(face, str(path)))
    pdfmetrics.registerFontFamily('regular', normal='regular', bold='bold', italic='regular', boldItalic='bold')
    script = read(HERE / 'script.json')
    warm = read(REPO / 'benchmarks/rtx5090-2026-10-24-r6.json')
    demo = read(HERE / 'demo/capture/provenance.json')
    expected = [(0, 20), (20, 50), (50, 85), (85, 110), (110, 150), (150, 175)]
    assert [(s['start_seconds'], s['end_seconds']) for s in script['segments']] == expected
    assert sha(HERE / 'demo/capture/kernel.ptx') == demo['ptx_sha256']
    slides = create_slides(warm, demo)
    pdf_path = output / 'slides.pdf'
    pdf = canvas.Canvas(str(pdf_path), pagesize=(WIDTH / 2, HEIGHT / 2), pageCompression=1)
    pdf.setTitle('Deuteron — 3 minute compiler presentation')
    pdf.setAuthor('Deuteron')
    for slide in slides:
        draw_pdf_slide(pdf, slide)
        pdf.showPage()
    pdf.save()
    with fitz.open(pdf_path) as document:
        assert len(document) == 6
        for index, page in enumerate(document):
            image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            assert (image.width, image.height) == (WIDTH, HEIGHT)
            image.save(frames / f'slide-{index + 1:02d}.png')
    build_pptx(slides, script, output / 'slides.pptx')
    word_count = make_script_documents(script, output)
    previews = Image.new('RGB', (1920, 1080), '#060C16')
    for index in range(6):
        with Image.open(frames / f'slide-{index + 1:02d}.png') as original:
            thumbnail = original.resize((620, 349), Image.Resampling.LANCZOS)
            previews.paste(thumbnail, (15 + (index % 3) * 635, 132 + (index // 3) * 432))
    draw = ImageDraw.Draw(previews)
    draw.text((22, 26), 'DEUTERON  /  THREE-MINUTE VIDEO', font=font(34, 'bold'), fill=INK)
    draw.text((24, 79), f'Six scenes · {word_count}-word narration · editable slides + 1080p frames', font=font(24), fill=MUTED)
    for index, segment in enumerate(script['segments']):
        label = f"{index + 1:02d}  {timestamp(segment['start_seconds'])}–{timestamp(segment['end_seconds'] if index < 5 else 180)}"
        draw.text((26 + (index % 3) * 635, 492 + (index // 3) * 432), label, font=font(24, 'bold'), fill=MINT)
    previews.save(output / 'slides-overview.png')
    source_files = [HERE / 'script.json', HERE / 'build_assets.py', HERE / 'requirements.txt',
                    HERE / 'demo/capture/kernel.ptx', HERE / 'demo/capture/provenance.json',
                    REPO / 'benchmarks/rtx5090-2026-10-24-r6.json', REPO / 'benchmarks/validation-2026-10-24-r6.json']
    artifacts = [pdf_path, output / 'slides.pptx', output / 'recording-script.pdf', output / 'recording-script.docx',
                 output / 'narration.txt', output / 'slides-overview.png', *sorted(frames.glob('slide-*.png'))]
    manifest = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'source_release': 'efb728d9106d864bb5f35efc95bfb5f01d95646a',
                'spoken_words': word_count, 'timeline_seconds': 180, 'narration_ends_seconds': 175,
                'slides': 6, 'frame_size': [WIDTH, HEIGHT], 'final_video_status': 'awaiting_user_recording',
                'source_sha256': {str(path.relative_to(REPO)): sha(path) for path in source_files},
                'artifacts': {str(path.relative_to(output)): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in artifacts},
                'font_files': {key: {'path': str(value), 'sha256': sha(value)} for key, value in FONT_FILES.items()},
                'layout': {'all_text_measured_with_build_fonts': True, 'all_text_within_canvas': True,
                           'pdf_and_png_share_rendering': True, 'pptx_uses_editable_text_and_shapes': True,
                           'pptx_rendering_note': 'PowerPoint rendering depends on installed fonts. PDF/PNG are the fixed visual exports.'}}
    (output / 'assets-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'slides': 6, 'spoken_words': word_count,
                      'files': len(artifacts), 'timeline_seconds': 180}, indent=2))


if __name__ == '__main__':
    main()
