"""Run: streamlit run app.py"""
import io
import pandas as pd
import streamlit as st
from analyst import analyze
from reporter import build_report

st.set_page_config(page_title='VeriQuery', page_icon='📊', layout='wide')
st.title('VeriQuery | Python Analyst & Reporter')
st.caption('Part F demo: CSV analysis. SQL and verification integration are available via the Python API.')
upload = st.file_uploader('Upload a CSV dataset', type=['csv'])
if upload is not None:
    try:
        raw = upload.getvalue()
        if len(raw) > 20_000_000:
            st.error('CSV exceeds the 20 MB demo limit.')
            st.stop()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        st.error(f'Could not read CSV: {exc}')
        st.stop()
    st.success(f'Loaded {len(df):,} rows and {len(df.columns)} columns')
    st.dataframe(df.head(20), use_container_width=True)
    st.write('**Columns:**', ', '.join(map(str, df.columns)))
    question = st.text_input('Ask a question', placeholder='e.g. What is the total sales by region?')
    if st.button('Analyze', type='primary') and question.strip():
        result = analyze(df, question)
        report = build_report(result)
        st.subheader('Answer')
        st.write(report['answer'])
        st.dataframe(report['table'], use_container_width=True)
        if report['figure'] is not None:
            st.plotly_chart(report['figure'], use_container_width=True)
        st.write('**Confidence:**', report['confidence'])
        st.write('**Verification:** Not connected in CSV-only demo')
        if report['assumptions']:
            st.warning('Assumptions: ' + '; '.join(report['assumptions']))
        if report['warnings']:
            st.error('Warnings: ' + '; '.join(report['warnings']))
        st.download_button('Download analysis CSV', report['table'].to_csv(index=False), 'analysis.csv', 'text/csv')
else:
    st.info('Upload a CSV to start.')
