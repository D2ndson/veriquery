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

p('VeriQuery SQL Writer Results','Title')
p('Harsh Saand  |  AI6130 Group Project  |  3 November 2026')
p('Fine-tuning Qwen2.5-Coder-7B-Instruct improved the observed execution accuracy from 36.36% to 40.40% on the same BIRD development subset. The gain is small: 13 questions were fixed, but 9 previously correct answers became wrong. I would keep this adapter as an experimental result rather than replace the team’s current SQL writer on this evidence alone.')
h('Measured results')
table(['Measure','Untuned 7B','QLoRA 7B'],[
('Correct answers among 99 scorable questions','36','40'),('Execution accuracy','36.36%','40.40%'),('Queries with execution errors','19','10'),('Executable but incorrect answers','44','49'),('Model calls across 100 questions','154','133'),('Mean writer time per question','6.25 s','7.91 s')],[3.8,1.55,1.55])
p('All 100 questions were attempted. Question 397 in card_games was excluded from accuracy because its reference result exceeded the 10,000-row limit. Counting correct answers over all attempts gives 36% and 40%, respectively.')
pic('outcomes.png')
cap('Figure 1  Outcomes for the same 100 questions. Each bar includes the one unscorable reference.')
p('The net improvement is 4.04 percentage points. Among the 99 paired scorable questions, 27 stayed correct and 50 stayed wrong. An exploratory exact paired test gives p = 0.523, so this small sample does not establish a reliable accuracy improvement.')
h('What this result covers')
p('This experiment isolates the 7B SQL writer with execution-error repair. It does not reproduce the friend’s reported Qwen2.5-Coder-32B failure rate, and it does not measure the integrated system’s confidence labels. The meeting’s target of 80–90% accuracy among high-confidence answers therefore remains untested here.')

h('Training and evaluation')
p('The starting checkpoint was Qwen2.5-Coder-7B-Instruct at revision c03e6d358207e414f1eca0bb1891e29f1db0e242. I used QLoRA so that training could run on one NTU A40 GPU. The base weights were loaded in 4-bit NF4 precision with double quantization; the trainable adapter had 40,370,176 parameters.')
table(['Setting','Value'],[('LoRA rank and scaling','16 and 32'),('Target modules','All linear layers'),('Learning rate and schedule','0.0001 with cosine decay'),('Batch and accumulation','1 example and 16 steps'),('Sequence limit and seed','4,096 tokens and 6130'),('Training completed','343 optimizer steps; 0.998 epochs')],[3.25,3.65])
p('The official filtered BIRD training release contains 6,601 examples. A database-disjoint split produced 6,039 training and 562 validation examples. After explicitly excluding 539 overlength training examples, 5,500 remained eligible for training. Evaluation databases were excluded from the training source. Only assistant SQL tokens contributed to the loss.')
pic('training_loss.png',6.2)
cap('Figure 2  Logged training loss and the final validation loss of 0.1085. Teacher-forced loss is not SQL execution accuracy.')
p('Training used official schema metadata and checked reference SQL syntax and safety; it did not execute the training references against full training databases. The configured one-epoch run reached 0.998 epochs because of the trainer’s batch accounting. The Slurm training job completed in 3 h 00 min 12 s, including validation and checkpoint saving.')
h('Matched comparison')
p('Both evaluations used the same fixed 100-question subset of the archived July 2025 Mini-Dev release, the same database files, 4-bit base loading, greedy decoding, a 16,384-token context, a 1,024-token output budget and at most two execution-error repairs. Queries had a five-second limit and a 10,000-row cap. Scoring compared sets of result rows, so row order and duplicate multiplicity were not part of the headline metric.')
p('The manifests match on every recorded field except the adapter and output paths. Dataset and adapter hashes were verified after download. Both evaluations ran on the A40 in the same Slurm job; their combined job time was 30 min 24 s.')

h('What changed in the queries')
p('The examples below explain the aggregate result. They were inspected after scoring and are illustrations, not a complete classification of all failures.')
D.add_heading('A repaired relationship in question 678',2)
p('For the most-viewed post by Harvey Motulsky, the untuned query filtered posts.OwnerDisplayName directly and returned the wrong result. The tuned query joined posts.OwnerUserId to users.Id and filtered users.DisplayName. That join matched the reference result.')
D.add_heading('A wrong join in question 825',2)
p('For the gender of the superhero with Phoenix Force, the original query correctly joined through superhero.gender_id. The tuned query instead matched gender.id directly to hero_power.hero_id. It executed successfully, but returned the wrong answer. Execution-error repair cannot detect this type of mistake on its own.')
D.add_heading('A changed aggregation in question 1039',2)
p('The original query averaged the player’s long_shots values and scored correct. The tuned query counted the values and divided by the number of distinct player identifiers. That changed the requested average and produced a regression. This is why I would not use lower loss or fewer SQL errors as evidence that the writer is ready.')
h('A practical novelty proposal')
p('The next experiment should use disagreement between SQL drafts to decide what to check. If drafts differ on a join, inspect the key relationship or row multiplicity; if they differ on an average, inspect the numerator, denominator and grouping level. Feed that evidence into one targeted repair and retain an uncertainty flag when the issue remains unresolved.')
pic('proposed_repair.png')
cap('Figure 3  Proposed extension. This workflow has not been implemented or evaluated in the present experiment.')
p('Candidate selection and verification are established ideas in CHASE-SQL and CHESS. The contribution to test here is whether a bounded, targeted check reduces wrong high-confidence answers compared with the team’s existing voting approach at the same model-call budget. Report accuracy together with high-confidence coverage, fixed cases, regressions and latency.')
h('Limits and recommendation')
p('This is one seed and one development subset, with one unscorable reference. There is no full-pipeline or 32B comparison. Keep the current team baseline, retain this adapter for the report, and use the observed join and aggregation failures to design the next controlled experiment. No claim of reaching the high-confidence target is supported by these results.')

h('Reproducibility and project files')
p('Training job 50403 and evaluation job 50407 both completed with exit code 0. The downloaded results include the final adapter, model and data manifests, question-level traces, paired outcomes, training logs and package versions. The small GitHub result bundle contains the recomputed summary, paired outcomes and evaluation manifests; the model weights and raw database rows stay in the local LLM folder.')
p('The public comparison can be inspected in code/experiments/sql_qlora_7b. Full local traces are in results/base7b/repair.jsonl and results/tuned7b/repair.jsonl. The final adapter is in results/qlora7b/adapter. Reproduction commands are in code/training/README.txt. The existing team subset was not changed; this experiment freezes its own 100 unique question IDs.')
h('Code improvements')
p('The update separates cached model responses by server endpoint and rejects responses cut off by the output-token limit. It also adds paired failure analysis, richer pipeline diagnostics and a reproducible training workflow. The local regression suite passed 211 tests before the report was added. These tests check software behaviour; the measured model results are the separate 100-question experiment above.')
h('GitHub change')
link('https://github.com/D2ndson/veriquery/pull/2')
p('The changes target part-a-integration. The pull request is open for team review; the integration branch has not been merged or overwritten.')
h('References')
link('https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct','Qwen team  Qwen2.5 Coder 7B Instruct model card')
link('https://huggingface.co/datasets/birdsql/bird23-train-filtered','BIRD team  Filtered BIRD training data  CC BY SA 4.0')
link('https://huggingface.co/datasets/birdsql/bird_mini_dev','BIRD team  Mini Dev data and archived releases')
link('https://huggingface.co/docs/peft/developer_guides/quantization','Hugging Face  PEFT quantization guidance')
link('https://arxiv.org/abs/2410.01943','Pourreza and colleagues  CHASE SQL  2024')
link('https://arxiv.org/abs/2405.16755','Talaei and colleagues  CHESS  2024')
p('The empirical figures in this report come from the saved NTU run records. Research references provide context for the proposed extension and are not evidence that the extension has been evaluated.')
# Page numbers only; no decorative header.
f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.CENTER
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)
# Apply black Times New Roman to all visible runs and table text.
for q in list(D.paragraphs)+[q for t in D.tables for row in t.rows for cell in row.cells for q in cell.paragraphs]:
    for run in q.runs:
        run.font.name='Times New Roman';run.font.color.rgb=RGBColor(0,0,0)
D.save(out/'VeriQuery_SQL_Writer_Report.docx')
print(out/'VeriQuery_SQL_Writer_Report.docx')
