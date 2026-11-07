"""Build the verified SQL experiment report using bundled python-docx."""
from pathlib import Path
import json,sys
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
root=Path(sys.argv[1]);out=root/'report';out.mkdir(exist_ok=True)
s=json.loads((root/'code/experiments/sql_qlora_7b/summary.json').read_text())
assert s['runs']['base7b']['correct']==36 and s['runs']['tuned7b']['correct']==40
D=Document();sec=D.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.8)
for name in ['Normal','Title','Heading 1','Heading 2','Caption']:
    style=D.styles[name];style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
    style.font.size=Pt(12 if name in ['Normal','Caption'] else 17 if name=='Title' else 14)
    style.paragraph_format.space_after=Pt(6)
D.styles['Normal'].paragraph_format.line_spacing=1.05
for st in D.styles:
    for border in list(st.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
D.styles['Heading 1'].paragraph_format.space_before=Pt(9)
D.styles['Heading 2'].font.size=Pt(12);D.styles['Heading 2'].font.bold=True
D.core_properties.author='Harsh Saand';D.core_properties.title='VeriQuery SQL Writer Results';D.core_properties.subject='Matched Qwen2.5 Coder 7B fine tuning experiment'

def p(text,style=None):return D.add_paragraph(text,style)
def h(text):D.add_heading(text,1)
def pic(name,width=6.8):
    q=D.add_paragraph();q.paragraph_format.space_after=Pt(3);q.add_run().add_picture(str(out/'figures'/name),width=Inches(width))
def cap(text):
    q=p(text,'Caption');q.paragraph_format.space_after=Pt(8)
def table(headers,rows,widths):
    t=D.add_table(rows=1, cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for i,(head,w) in enumerate(zip(headers,widths)):t.columns[i].width=Inches(w);t.rows[0].cells[i].text=head
    for row in rows:
        for cell,text in zip(t.add_row().cells,row):cell.text=str(text)
    for ri,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr();no=OxmlElement('w:cantSplit');trpr.append(no)
        for ci,cell in enumerate(row.cells):
            cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right']:
                e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
            pr.append(b);m=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                e=OxmlElement('w:'+side);e.set(qn('w:w'),'70');e.set(qn('w:type'),'dxa');m.append(e)
            pr.append(m)
            if ri==0:
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EFEFEF');pr.append(sh)
            for q in cell.paragraphs:
                q.paragraph_format.space_after=Pt(1);q.paragraph_format.space_before=Pt(1)
                if ci:q.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in q.runs:run.font.bold=ri==0;run.font.size=Pt(12)
    flag=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(flag)
    return t

def link(url,label=None):
    q=D.add_paragraph();q.paragraph_format.space_after=Pt(4)
    rel=q.part.relate_to(url,RT.HYPERLINK,is_external=True);hyper=OxmlElement('w:hyperlink');hyper.set(qn('r:id'),rel)
    run=OxmlElement('w:r');pr=OxmlElement('w:rPr');fonts=OxmlElement('w:rFonts');fonts.set(qn('w:ascii'),'Times New Roman');fonts.set(qn('w:hAnsi'),'Times New Roman');pr.append(fonts)
    c=OxmlElement('w:color');c.set(qn('w:val'),'000000');pr.append(c);run.append(pr);txt=OxmlElement('w:t');txt.text=label or url;run.append(txt);hyper.append(run);q._p.append(hyper)

p('VeriQuery SQL Writer Evaluation','Title')
p('Harsh Saand  |  AI6130 Group Project')
p('Fine-tuning Qwen2.5-Coder-7B-Instruct increased execution accuracy from 36.36% to 40.40% on a matched BIRD evaluation. The improvement was modest, with 13 fixes and 9 regressions.')
h('Model and fine tuning')
p('I trained a QLoRA adapter on 5,500 BIRD question–SQL examples, with 562 validation examples from separate databases. Training used 4-bit NF4 loading, LoRA rank 16, scaling 32, a learning rate of 0.0001 and a 4,096-token sequence limit. The run completed 343 optimizer steps, approximately one epoch, on an NTU NVIDIA A40 in about three hours.')
h('Evaluation')
p('The original and fine-tuned 7B models answered the same 100 unique MiniDev questions with identical prompts, greedy decoding and up to two execution-error repairs. One reference answer exceeded the 10,000-row limit, leaving 99 scorable questions. Accuracy compares sets of returned rows.')
table(['Measure','Original 7B','Fine tuned 7B'],[
('Correct answers','36 of 99','40 of 99'),('Execution accuracy','36.36%','40.40%'),('Execution errors','19','10'),('Executable wrong answers','44','49'),('Mean writer time per question','6.25 s','7.91 s')],[3.8,1.55,1.55])
pic('outcomes.png',6.2)
cap('Outcomes across all 100 questions, including one unscorable reference.')
h('Conclusion')
p('The gain was 4.04 percentage points. Fine-tuning reduced execution errors, but some incorrect queries still ran successfully. The paired exact test gave p = 0.523, so this sample does not establish a reliable improvement. These results cover the 7B SQL writer; they do not measure the 32B model or high-confidence accuracy.')
# Page numbers only; no decorative header.
f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.CENTER
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)
# Apply black Times New Roman to all visible runs and table text.
for q in list(D.paragraphs)+[q for t in D.tables for row in t.rows for cell in row.cells for q in cell.paragraphs]:
    for run in q.runs:
        run.font.name='Times New Roman';run.font.color.rgb=RGBColor(0,0,0)
D.save(out/'VeriQuery_SQL_Writer_Report.docx')
print(out/'VeriQuery_SQL_Writer_Report.docx')
