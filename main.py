import io
from coverage import Coverage
import streamlit as st
import pandas as pd
from simulate_data import Sampling
import plotly.graph_objects as go
import time
import math
import numpy as np
import re
from PIL import Image
from warnings import filterwarnings
filterwarnings("ignore")

INFLOWS_LOGO_PATH = "images/InFLOWS-logo-white_font-1-1024x458.png"
DATA_FORMAT_EXAMPLE_PATH = "data_format_example.xlsx"
RUNS_COUNT_FILE_PATH = "runs_count.txt"

def get_df_from_uploaded_file(file) -> pd.DataFrame :
    try :
        return pd.read_csv(file)
    except ValueError:
        return pd.read_excel(file)

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True)
    processed_data = output.getvalue()
    return processed_data

def typewriter(text, delay=0.1):
    output_text = ""
    placeholder = st.empty()
    for char in text:
        output_text += char
        placeholder.markdown(output_text, unsafe_allow_html=True)
        time.sleep(delay)

logical_symbols_dict = {"==(Equal to)": "==",
                        "!=(Not Equal to)": "!=",
                        ">(Greater than)": ">",
                        ">=(Greater or Equal)": ">=",
                        "<(Less than)": "<",
                        "<=(Less or Equal)": "<=",
                        "Mixture Size ==": "==",
                        "Mixture Size <=": "<=",
                        "Mixture Size >=": ">="}


# MAIN PAGE
st.set_page_config(page_title="Coverage Design Flow")

with open(RUNS_COUNT_FILE_PATH, "r") as runs_file:
    runs_count = int(runs_file.read())

st.header("Space - Filling Analyzer")
st.write("Assess how well your experiments explore the design space", delay=0.02)

with st.sidebar.container():

    col1, im, col3 = st.columns(3)
    inflows_logo = Image.open(INFLOWS_LOGO_PATH)
    with im:
        st.image(inflows_logo)
    st.write('')
    st.write('')

    with st.expander(label="Data upload", expanded=True):
        all_columns = []
        ui_samples_data = st.file_uploader(label="Import samples data", type=["csv", "xlsx"])
        sample_data_format = pd.read_excel(DATA_FORMAT_EXAMPLE_PATH, index_col="ID")

        # DATA FORMAT SAMPLE
        if not ui_samples_data :
            with st.popover('See data format example', use_container_width=True):
                st.dataframe(sample_data_format)
                excel_data = to_excel(sample_data_format)
                st.download_button(
                    label="Download as excel",
                    data=excel_data,
                    file_name="Data_format_example.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # LOAD USER DATA
        if ui_samples_data:
            ui_df = get_df_from_uploaded_file(ui_samples_data)
            # Check ID column
            if "ID" not in ui_df.columns :
                ui_df_shape = ui_df.shape[0]
                ui_df_index = ["type", "Min", "Max"] + list(range(1,ui_df_shape-2,1))
                ui_df['ID'] = ui_df_index
            ui_df.index = ui_df.ID
            ui_df = ui_df.drop('ID', axis=1)
            ui_df.fillna('', inplace=True)

            # Gestion message toast
            if 'toast_shown' not in st.session_state:
                st.session_state.toast_shown = False

            # Limit ui data columns to 30 and rows to 150 max
            if ui_df.shape[0] > 150 or ui_df.shape[1] > 30:
                st.error("⚠️ Input data too large\nMax size is 150x30")
                st.stop()

            # Check data format
            try :
                ui_df.loc['Min':, :] = ui_df.loc['Min':, :].map(eval)
            except TypeError:
                pass
            sub_df_to_check = ui_df.loc['Min':, :]
            if not sub_df_to_check.map(np.isreal).all().all() :
                st.error("⚠️ Invalid data format")
                st.stop()

            all_columns = ui_df.columns
            st.dataframe(ui_df)

        ui_masked_cols = st.multiselect(label="Columns to be ignored", options=all_columns, disabled=len(all_columns)==0, key="multiselect1")

    total_weight = st.number_input(label="Total weight (g)", min_value=0, step=10, placeholder=100, value=100, disabled=not ui_samples_data)

    with st.expander(label="Space-Filling Constraints", expanded=True):
        col_equation, col_logical_test, col_numerical = st.columns(3)
        with col_equation :
            ui_equation = st.text_input(label="Equation", disabled=not ui_samples_data)
        with col_logical_test :
            ui_logical_test = st.selectbox(label="Logical test", options=["==(Equal to)", "!=(Not Equal to)", ">(Greater than)", ">=(Greater or Equal)",
                                                                          "<(Less than)", "<=(Less or Equal)", "Mixture Size ==", "Mixture Size <=",
                                                                          "Mixture Size >="],
                                           index=None,
                                           placeholder="Choose an option",
                                           disabled=not ui_samples_data)
        with col_numerical :
            ui_numerical = st.number_input(label="Numerical", step=1, disabled=not ui_samples_data)

    with st.expander(label="Advanced parameters", expanded=False) :
        ui_threshold = float(st.number_input(label="Threshold", disabled=not ui_samples_data, min_value=0.00, value=.25))

    mixture_len_constraint = []
    minmax_space_constraint = []
    #if ui_logical_test.startswith('Mixture') :
    #    mixture_len_constraint = [ui_equation+logical_symbols_dict[ui_logical_test]+str(ui_numerical)]
    #else :
    #    minmax_space_constraint = [ui_equation+logical_symbols_dict[ui_logical_test]+str(ui_numerical)]

    compute_coverage_btn = st.button("Run coverage", disabled=not ui_samples_data)

if compute_coverage_btn :
    desc_dict = ui_df.loc['type':'Max', :].to_dict(orient="dict")
    data_sampler = Sampling(minmax_space_constraints=minmax_space_constraint,
                            mixture_len_constraints=mixture_len_constraint,
                            total_weight=total_weight,
                            desc_dict=desc_dict)
    simulated_df = data_sampler.get_samples()
    simulated_df.drop(ui_masked_cols, axis=1, inplace=True)

    ui_df = ui_df.loc[1:, :]

    coverage_object = Coverage(ui_df, simulated_df)
    coverage_value = coverage_object.get_coverage(threshold=ui_threshold)

    # Plot coverage
    coverage_side = (coverage_value/100) ** 0.5

    n_grid_lines = 10

    base_square = go.Scatter(
        x=[0, 1, 1, 0, 0],
        y=[0, 0, 1, 1, 0],
        fill="toself",
        fillcolor="lightgray",
        line=dict(color="black"),
        name="Design Space",
        hovertemplate=f"Coverage: {coverage_value} %<extra></extra>"
    )

    coverage_square = go.Scatter(
        x=[0, coverage_side, coverage_side, 0, 0],
        y=[0, 0, coverage_side, coverage_side, 0],
        mode="lines",
        fill="toself",
        fillcolor="rgba(0, 200, 20, .5)",
        line=dict(width=0, color="rgba(0, 200, 20, .5)"),
        name="Data coverage",
        hovertemplate="Data coverage"
    )

    # Grille interne
    grid_lines = []
    step = 1 / n_grid_lines
    for i in range(1, n_grid_lines):
        # lignes verticales
        grid_lines.append(go.Scatter(
            x=[i * step, i * step],
            y=[0, 1],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        # lignes horizontales
        grid_lines.append(go.Scatter(
            x=[0, 1],
            y=[i * step, i * step],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))

#Design Space coverage
    fig = go.Figure([base_square, coverage_square]+grid_lines)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        showlegend=True,
        width=500,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_scaleanchor="x",
        legend=dict(
            font=dict(size=20),
            y=.975,
            x=1.2
        )
    )

    with st.spinner("Computing coverage...", show_time=False):
        time.sleep(2)
    with st.spinner("Drawing graphic...", show_time=False):
        time.sleep(2)
    st.success(f"COVERAGE: {coverage_value} %")
    st.plotly_chart(fig)

    #multiplier_temp = math.ceil(90/coverage_value) if coverage_value != 0 else 100
    #if multiplier_temp * coverage_value > 100 :
    #    multiplier = math.floor(90 / coverage_value)
    #else :
    #    multiplier = multiplier_temp

    typewriter(f"""
        <p><em>Wondering which experiments and how many you need to 10X your coverage ?</em><br>
        🗓️<em> Book a 15-minutes chat with our expert 👉 <a href='https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ1UrQ85a9zX1YOTkTkGgLcaz2Pb5-2cDKGsrGHd30-IeXZ376Ux7yR_vDTpiWJzlN8LGgTXCQ0J' target="_blank">here</a>
        </em></p>
        """, delay=.02)

    runs_count += 1

# TOTAL RUNS
st.markdown(f"""
    <style>
    .custom-footer {{
        position: fixed;
        bottom: 20px;
        left: 0;
        width: 100%;
        text-align: center;
        z-index: 9999;
    }}

    .footer-box {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 20px;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        font-size: 20px;
        margin-left: 10%;
    }}

    .footer-input {{
        margin-left: 10px;
        padding: 5px 10px;
        font-size: 20px;
        width: 160px;
        border: 1px solid #ccc;
        border-radius: 4px;
    }}
    </style>

    <div class="custom-footer">
        <div class="footer-box">
            <span><strong>TOTAL RUNS:</strong></span>
            <input class="footer-input" type="text" value="{runs_count}" readonly />
        </div>
    </div>
""", unsafe_allow_html=True)

with open(RUNS_COUNT_FILE_PATH, "w") as runs_file :
    runs_file.write(str(runs_count))
