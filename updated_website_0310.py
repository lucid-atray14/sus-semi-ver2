import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
import streamlit as st
import numpy as np
from pymcdm.methods import PROMETHEE_II
from pymcdm.helpers import rankdata
from pymcdm.methods import TOPSIS
from pymcdm.weights import entropy_weights
from io import BytesIO
from bokeh.palettes import Category10
from bokeh.models import NumeralTickFormatter
from bokeh.models import LogScale, Range1d, LinearScale
from scipy.stats import ks_2samp
from bokeh.transform import jitter, factor_cmap

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        /* Lighten sidebar background */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Dark text for light sidebar */
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }
        
        /* Sidebar hover effects */
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# ===== CACHED FUNCTIONS =====
@st.cache_data
def load_final_database():
    df_original = pd.read_excel("final_database.xlsx")
    return df_original.iloc[:, 1:]

@st.cache_data
def load_bandgap_database():
    df1_original = pd.read_excel("bandgap_database.xlsx")
    return df1_original.iloc[:, 1:]

@st.cache_data
def filter_dataframe(_df, filters, selected_names=None):
    """Filter dataframe based on provided filters and optional names"""
    filtered = _df.copy()
    
    # Apply each filter dynamically
    for filter_name, filter_range in filters.items():
        if filter_name in _df.columns:
            filtered = filtered[
                filtered[filter_name].between(filter_range[0], filter_range[1], inclusive='both')
            ]
    
    if selected_names is not None:
        filtered = filtered[filtered["Name"].isin(selected_names)]
    
    return filtered

@st.cache_data
def calculate_weights(matrix, method="entropy"):
    if method == "entropy":
        return entropy_weights(matrix)
    return None

@st.cache_data
def run_topsis(matrix, weights, criteria_types):
    topsis = TOPSIS()
    return topsis(matrix, weights, criteria_types)

@st.cache_data
def run_promethee(matrix, weights, criteria_types):
    promethee = PROMETHEE_II('usual')
    return promethee(matrix, weights, criteria_types)

@st.cache_data
def prepare_plot_data(df, x_col, y_col, log_x=False, log_y=False):
    df_plot = df.copy()
    if log_x:
        df_plot[x_col] = np.log10(df_plot[x_col].clip(lower=1e-10))
    if log_y:
        df_plot[y_col] = np.log10(df_plot[y_col].clip(lower=1e-10))
    return df_plot

@st.cache_data
def create_full_output(filtered_df, results_df, weights_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        full_data = filtered_df.copy()
        if 'Score' in results_df.columns:
            full_data['TOPSIS_Score'] = results_df['Score']
            full_data['TOPSIS_Rank'] = results_df['Rank']
        else:
            full_data['PROMETHEE_Net_Flow'] = results_df['Net Flow']
            full_data['PROMETHEE_Rank'] = results_df['Rank']
        full_data.to_excel(writer, sheet_name='Full Data', index=False)
        results_df.to_excel(writer, sheet_name='Rankings', index=False)
        weights_df.to_excel(writer, sheet_name='Weights', index=False)
        pd.DataFrame.from_dict(st.session_state.filters, orient='index').to_excel(
            writer, sheet_name='Filter Settings'
        )
    return output.getvalue()

def create_professional_plot(df, x_col, y_col, title, x_label, y_label, log_x=False, log_y=False):
    # Create a copy to avoid modifying the original dataframe
    df_plot = df.copy()
    
    # Professional color palette
    primary_color = "#3498db"
    highlight_color = "#c5301f"
    
    # Create the figure with dynamic axis types
    p = figure(
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        x_axis_label=f"log({x_label})" if log_x else x_label,
        y_axis_label=f"log({y_label})" if log_y else y_label,
        x_axis_type="log" if log_x else "linear",
        y_axis_type="log" if log_y else "linear",
        width=800,
        height=500,
        tooltips=[("Name", "@Name")],
        toolbar_location="above",
        sizing_mode="stretch_width"
    )
    
    # Handle negative/zero values for log scales
    if log_x:
        df_plot[x_col] = df_plot[x_col].clip(lower=1e-10)
    if log_y:
        df_plot[y_col] = df_plot[y_col].clip(lower=1e-10)
    
    # Plot all points
    source = ColumnDataSource(df_plot)
    p.circle(
        x=x_col,
        y=y_col,
        source=source,
        size=8,
        color=primary_color,
        alpha=0.6,
        legend_label="All Materials"
    )
    
    # Highlight exactly 10 random materials
    num_highlight = min(10, len(df_plot))
    highlight_df = df_plot.sample(n=num_highlight, random_state=42)
    highlight_source = ColumnDataSource(highlight_df)
    
    p.circle(
        x=x_col,
        y=y_col,
        source=highlight_source,
        size=12,
        color=highlight_color,
        alpha=1.0,
        legend_label="Highlighted Materials"
    )
    
    # Add labels to highlighted points
    labels = LabelSet(
        x=x_col,
        y=y_col,
        text="Name",
        source=highlight_source,
        text_font_size="10pt",
        text_color=highlight_color,
        y_offset=8,
        text_align='center'
    )
    p.add_layout(labels)
    
    # Professional legend styling
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.7
    p.legend.label_text_font_size = "12pt"
    
    # Grid and axis styling
    p.xgrid.grid_line_color = "#e0e0e0"
    p.ygrid.grid_line_color = "#e0e0e0"
    p.axis.minor_tick_line_color = None
    
    return p

def main():
    set_custom_style()
    df = load_final_database()
    df1 = load_bandgap_database()
    
    # Sidebar navigation
    st.sidebar.title("📊 Material Analysis")
    st.sidebar.markdown("---")
    selected_page = st.sidebar.radio(
        "Navigation Menu", 
        ["Home", "Bandgap Information", "Decision-making Assistant"],
        captions=["Welcome page", "Commonly researched semiconductors", "Multi-criteria decision making tool"]
    )
    
    # Add footer
    st.markdown("""
    <div class="footer">
        Semiconductor Database © 2025 | v2.0 | Developed by HERAWS
    </div>
    """, unsafe_allow_html=True)

    if selected_page == "Home":
        st.title("Semiconductor Database")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            ### 🔍 About This Tool
            This interactive platform enables comprehensive analysis of environmental impacts and sustainability of semiconductors with:
            - **Extensive database** on ESG scores, CO₂ footprints, and more
            - **Visualizations** to explore relationships between parameters
            - **Multi-criteria** decision making tools (TOPSIS, PROMETHEE)
            - **Export capabilities** for further analysis
            """)
            
        with cols[1]:
            st.markdown("""
            ### 🚀 Getting Started
            1. Select an analysis page from the sidebar
            2. Configure your filters and parameters
            3. Visualize the relationships
            4. Download results for further use
            
            **Pro Tip:** Use the MCDM analysis for ranking the most promising semiconductors.
            """)
        
        st.markdown("---")
        
        with st.expander("📚 Database Information", expanded=True):
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Materials", len(df))
            with cols[1]:
                st.metric("Bandgap Range", f"{df['Bandgap'].min():.1f} - {df['Bandgap'].max():.1f} eV")
            with cols[2]:
                st.metric("Production Range", f"{df['Production (ton)'].min():.1f} - {df['Production (ton)'].max():.1f} tons")
        
    elif selected_page == "Bandgap Information":
        st.title("Bandgap Information")
        st.markdown("Most commonly researched semiconductors and their band gap range.")
    
        # Filters section at the top in expandable containers
        with st.expander("🔍 Filter Settings", expanded=True):
            cols = st.columns(2)
            
            with cols[0]:
                st.markdown("**Material Properties**")
                y_col = st.selectbox(
                    "Y-Axis Property", 
                    [col for col in df1.columns if col not in ['Name', 'Bandgap']],
                    help="Select the property to plot against bandgap"
                )
    
            with cols[1]:
                st.markdown("**Material Selection**")
            
                specified_names = [
                    "TiO2","ZnO","CdS","MoS2","SnO2","ZnS","WO3","CuO","Cu2O","Si"
                ]
                selected_names = st.multiselect(
                    "Select specific materials",
                    specified_names,
                    default=["TiO2", "ZnO"],
                    help="Focus on specific materials of interest"
                )
    
        # Process data with distinct colors
        custom_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", 
                         "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5"]
        name_colors = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(specified_names)}
        
        df1['color'] = df1['Name'].map(name_colors)
        filtered_df = df1[df1['Name'].isin(selected_names)] if selected_names else df1
        
        # Calculate median for each material
        median_data = filtered_df.groupby('Name')['Bandgap'].median().reset_index()
        median_data.columns = ['Name', 'Median_Bandgap']
        median_data['color'] = median_data['Name'].map(name_colors)
        
        # Plot section below filters
        st.markdown("---")
        st.markdown(f"**Analysis Results ({len(filtered_df)} materials)**")
    
        source = ColumnDataSource(filtered_df)
        
        # --- Plot setup ---
        p = figure(
            x_range=filtered_df["Name"].unique().tolist(),
            width=600, height=500,
            toolbar_location=None,
            title=None
        )
        
        # --- Scatter points with jitter and color mapping ---
        p.circle(
            x=jitter("Name", width=0.3, range=p.x_range),
            y="Bandgap",
            source=source,
            size=10, 
            alpha=0.9,
            color="color",
            legend_field="Name"
        )
        
        # --- Add median lines for each material ---
        for _, row in median_data.iterrows():
            p.line(
                x=[row['Name'], row['Name']],
                y=[row['Median_Bandgap'], row['Median_Bandgap']],
                line_width=40,
                line_color=row['color'],
                alpha=0.7,
                line_cap="round"
            )
        
        # --- Hover tool ---
        hover = HoverTool(
            tooltips=[
                ("Material", "@Name"),
                ("Bandgap (eV)", "@Bandgap")
            ]
        )
        p.add_tools(hover)
        
        # --- Legend configuration ---
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        
        # --- Aesthetics ---
        p.xaxis.axis_label = "Semiconductor"
        p.yaxis.axis_label = "Bandgap (eV)"
        p.xgrid.visible = False
        p.ygrid.visible = True
        p.outline_line_color = None
        
        st.bokeh_chart(p, use_container_width=True)
        
        # Display median statistics
        st.markdown("### Median Bandgap Values")
        median_display = median_data[['Name', 'Median_Bandgap']].copy()
        median_display.columns = ['Material', 'Median Bandgap (eV)']
        st.dataframe(median_display, use_container_width=True, hide_index=True)
        
        # Data download
        st.download_button(
            label="📥 Download Analysis Data",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="bandgap_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    elif selected_page == "Decision-making Assistant":
        st.title("Decision-making Assistant")
        st.markdown("Facilitate semiconductor selection with advanced filtering and visualization")
        
        # Initialize session state
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
        if 'show_additional_filters' not in st.session_state:
            st.session_state.show_additional_filters = False
        if 'initial_filter_name' not in st.session_state:
            st.session_state.initial_filter_name = None
        if 'second_graph_filters' not in st.session_state:
            st.session_state.second_graph_filters = {}
        if 'initial_filters_only' not in st.session_state:
            st.session_state.initial_filters_only = {}
        
        with st.expander("🔧 Filter Settings", expanded=True):
            cols = st.columns(2)
            
            with cols[0]:
                st.subheader("Bandgap Selection")
                col1, col2 = st.columns(2)
                with col1:
                    bandgap_min = st.number_input(
                        "Min (eV)",
                        min_value=0.0,
                        max_value=20.0,
                        value=0.0,
                        step=0.1,
                        key="bandgap_min"
                    )
                with col2:
                    bandgap_max = st.number_input(
                        "Max (eV)",
                        min_value=0.0,
                        max_value=20.0,
                        value=3.0,
                        step=0.1,
                        key="bandgap_max"
                    )

                # Optional: Add validation to ensure min <= max
                if bandgap_min > bandgap_max:
                    st.error("Minimum bandgap must be less than or equal to maximum bandgap")
                    
                bandgap_range = (bandgap_min, bandgap_max)
            
            with cols[1]:
                st.subheader("Additional Filter")
                filter_options = [
                    'Reserve (ton)', 'Production (ton)', 'HHI (USGS)',
                    'ESG Score', 'CO2 footprint max (kg/kg)', 
                    'Embodied energy max (MJ/kg)', 'Water usage max (l/kg)', 
                    'Toxicity', 'Companionality'
                ]
                
                selected_filter = st.selectbox("Choose a filter", filter_options, key="selected_filter")
                
                # Show slider only after a filter is selected
                if selected_filter:
                    # Get min and max values for the selected filter
                    filter_min = float(df[selected_filter].min())
                    filter_max = float(df[selected_filter].max())
                    
                    # Integer slider for Toxicity
                    if selected_filter == 'Toxicity':
                        filter_range = st.slider(
                            f"{selected_filter} Range",
                            int(filter_min),
                            int(filter_max),
                            (int(filter_min), int(filter_max)),
                            step=1,
                            key="initial_filter_slider"
                        )
                    else:
                        filter_range = st.slider(
                            f"{selected_filter} Range",
                            filter_min,
                            filter_max,
                            (filter_min, filter_max),
                            key="initial_filter_slider"
                        )
                else:
                    filter_range = None
            
            if st.button("Apply Initial Filters", key="apply_initial_filter"):
                if filter_range is not None:
                    # Clear previous filters and reset
                    st.session_state.filters = {
                        "Bandgap": bandgap_range,
                        selected_filter: filter_range
                    }
                    st.session_state.initial_filters_only = {
                        "Bandgap": bandgap_range,
                        selected_filter: filter_range
                    }
                    st.session_state.initial_filter_name = selected_filter
                    st.session_state.show_additional_filters = True
                    st.session_state.show_second_graph = False  # Reset second graph
                    st.session_state.second_graph_filters = {}
                    st.success("Initial filters applied!")
                    st.rerun()
                else:
                    st.warning("Please select an additional filter and set its range.")
        
        if st.session_state.filters:  # Check if any filters exist
            # Plot configuration - first graph is always Bandgap vs selected filter
            st.subheader("📊 Plot Configuration")
            
            # Get the selected filter from session state (the one that was chosen initially)
            initial_filters = [k for k in st.session_state.filters.keys() if k != 'Bandgap']
            
            if initial_filters:
                # First plot: Bandgap vs the initially selected filter
                x_col = 'Bandgap'
                y_col = initial_filters[0]
                
                st.info(f"Initial plot: **Bandgap** vs **{y_col}**")
                
                log_x = False  # Bandgap is never on log scale
                log_y = st.checkbox(f"Log scale Y-axis ({y_col})")
            else:
                # Fallback if somehow no initial filter was selected
                cols = st.columns(2)
                with cols[0]:
                    x_col = st.selectbox("X-Axis", [col for col in df.columns if col != 'Name'])
                    log_x = st.checkbox(f"Log scale X-axis")
                with cols[1]:
                    y_col = st.selectbox("Y-Axis", [col for col in df.columns if col not in ['Name', x_col]])
                    log_y = st.checkbox(f"Log scale Y-axis")
            
            # Apply filters and create plot - use only initial filters for first graph
            filtered_df = filter_dataframe(df, st.session_state.initial_filters_only)
            
            if not filtered_df.empty:
                st.success(f"🔄 {len(filtered_df)} materials match current filters")
                
                # Advanced options
                with st.expander("🎨 Customization Options"):
                    plot_title = st.text_input("Plot Title", f"{x_col} vs {y_col}")
                
                # Create and display plot
                p = create_professional_plot(
                    filtered_df, x_col, y_col, plot_title, x_col, y_col, log_x, log_y
                )
                st.bokeh_chart(p, use_container_width=True)
                
                # Show additional filters after graph is drawn
                if st.session_state.show_additional_filters:
                    st.markdown("---")
                    with st.expander("🔧 Choose 2 Additional Filters for Second Graph", expanded=True):
                        st.info("Select 2 filters to create axes for the second graph. Results from the first graph will be used.")
                        
                        # Get all available filters
                        all_filters = [
                            'Reserve (ton)', 'Production (ton)', 'HHI (USGS)',
                            'ESG Score', 'CO2 footprint max (kg/kg)', 
                            'Embodied energy max (MJ/kg)', 'Water usage max (l/kg)', 
                            'Toxicity', 'Companionality'
                        ]
                        
                        # Only exclude the initial filter that was already applied, keep everything else available
                        available_filters = [f for f in all_filters if f != st.session_state.initial_filter_name]
                        
                        # Dropdowns outside form to allow dynamic updates
                        cols = st.columns(2)
                        
                        with cols[0]:
                            filter_1 = st.selectbox("First filter (X-axis)", available_filters, key="filter_1_select")
                        
                        with cols[1]:
                            # Exclude filter_1 from second dropdown
                            available_for_filter_2 = [f for f in available_filters if f != filter_1]
                            filter_2 = st.selectbox("Second filter (Y-axis)", available_for_filter_2, key="filter_2_select")
                        
                        # Form for sliders only - prevents reload on slider changes
                        with st.form(key="second_graph_form"):
                            cols2 = st.columns(2)
                            
                            with cols2[0]:
                                if filter_1:
                                    filter_1_min = float(df[filter_1].min())
                                    filter_1_max = float(df[filter_1].max())
                                    
                                    # Integer slider for Toxicity
                                    if filter_1 == 'Toxicity':
                                        filter_1_range = st.slider(
                                            f"{filter_1} Range",
                                            int(filter_1_min),
                                            int(filter_1_max),
                                            (int(filter_1_min), int(filter_1_max)),
                                            step=1,
                                            key="filter_1_slider_form"
                                        )
                                    else:
                                        filter_1_range = st.slider(
                                            f"{filter_1} Range",
                                            filter_1_min,
                                            filter_1_max,
                                            (filter_1_min, filter_1_max),
                                            key="filter_1_slider_form"
                                        )
                            
                            with cols2[1]:
                                if filter_2:
                                    filter_2_min = float(df[filter_2].min())
                                    filter_2_max = float(df[filter_2].max())
                                    
                                    # Integer slider for Toxicity
                                    if filter_2 == 'Toxicity':
                                        filter_2_range = st.slider(
                                            f"{filter_2} Range",
                                            int(filter_2_min),
                                            int(filter_2_max),
                                            (int(filter_2_min), int(filter_2_max)),
                                            step=1,
                                            key="filter_2_slider_form"
                                        )
                                    else:
                                        filter_2_range = st.slider(
                                            f"{filter_2} Range",
                                            filter_2_min,
                                            filter_2_max,
                                            (filter_2_min, filter_2_max),
                                            key="filter_2_slider_form"
                                        )
                            
                            # Submit button for the form
                            submit_button = st.form_submit_button("Create Second Graph")
                        
                        if submit_button and filter_1 and filter_2:
                            # Store second graph filters separately
                            st.session_state.second_graph_filters = {
                                filter_1: filter_1_range,
                                filter_2: filter_2_range
                            }
                            
                            # Combine initial filters with second graph filters for the second graph only
                            st.session_state.filters = {**st.session_state.initial_filters_only, **st.session_state.second_graph_filters}
                            
                            st.session_state.show_second_graph = True
                            st.session_state.second_graph_x = filter_1
                            st.session_state.second_graph_y = filter_2
                            st.success("Second graph filters applied!")
                            st.rerun()
                    
                    # Second Graph - only show after filters are chosen
                    if 'show_second_graph' in st.session_state and st.session_state.show_second_graph:
                        st.markdown("---")
                        st.subheader("📊 Second Graph")
                        
                        x_col_2 = st.session_state.second_graph_x
                        y_col_2 = st.session_state.second_graph_y
                        
                        # Re-filter the data with all applied filters (from first graph + two new filters)
                        filtered_df_2 = filter_dataframe(df, st.session_state.filters)
                        
                        if not filtered_df_2.empty:
                            st.success(f"🔄 {len(filtered_df_2)} materials match all filters")
                            
                            cols = st.columns(2)
                            with cols[0]:
                                log_x_2 = st.checkbox(f"Log scale X-axis ({x_col_2})", key="log_x_2")
                            with cols[1]:
                                log_y_2 = st.checkbox(f"Log scale Y-axis ({y_col_2})", key="log_y_2")
                            
                            with st.expander("🎨 Customization Options - Second Graph"):
                                plot_title_2 = st.text_input("Plot Title", f"{x_col_2} vs {y_col_2}", key="plot_title_2")
                            
                            # Create and display plot
                            p2 = create_professional_plot(
                                filtered_df_2, x_col_2, y_col_2, plot_title_2, x_col_2, y_col_2, log_x_2, log_y_2
                            )
                            st.bokeh_chart(p2, use_container_width=True)
                            
                            # Data table for second graph
                            with st.expander("📋 View Data - Second Graph"):
                                st.dataframe(filtered_df_2[[x_col_2, y_col_2, "Name"]].sort_values(y_col_2, ascending=False))
                            
                            # MCDM Analysis Section
                            st.markdown("---")
                            st.subheader("📊 Multi-Criteria Decision Making")
                            st.info(f"Analyze the {len(filtered_df_2)} filtered materials using TOPSIS or PROMETHEE methods")
                            
                            cols_mcdm = st.columns(2)
                            with cols_mcdm[0]:
                                mcdm_method = st.selectbox(
                                    "Method",
                                    ["TOPSIS", "PROMETHEE"],
                                    help="TOPSIS: Technique for Order Preference by Similarity to Ideal Solution\nPROMETHEE: Preference Ranking Organization Method for Enrichment Evaluation",
                                    key="mcdm_method_custom"
                                )
                            with cols_mcdm[1]:
                                if mcdm_method == "TOPSIS":
                                    weighting_method = st.radio(
                                        "Weighting",
                                        ["Entropy Weighting", "Manual Weights"],
                                        horizontal=True,
                                        key="mcdm_weighting_custom"
                                    )
                            
                            # Criteria selection
                            criteria_options = {
                                'Reserve (ton)': 1, 'Production (ton)': 1, 'HHI (USGS)': -1,
                                'ESG Score': -1, 'CO2 footprint max (kg/kg)': -1,
                                'Embodied energy max (MJ/kg)': -1, 'Water usage max (l/kg)': -1,
                                'Toxicity': -1, 'Companionality': -1
                            }
                            available_criteria = {k: v for k, v in criteria_options.items() if k in filtered_df_2.columns}
                            
                            # Weight assignment
                            if mcdm_method == "TOPSIS" and weighting_method == "Entropy Weighting":
                                weights = entropy_weights(filtered_df_2[list(available_criteria.keys())].values)
                            else:
                                st.markdown("**📊 Criteria Weights** - Assign importance (0–5 scale):")
                                
                                weights = []
                                cols_weights = st.columns(len(available_criteria))
                                for i, (col, direction) in enumerate(available_criteria.items()):
                                    with cols_weights[i]:
                                        weight = st.slider(
                                            f"{col} ({'Max' if direction == 1 else 'Min'})",
                                            0, 5, 3,
                                            key=f"weight_custom_{col}"
                                        )
                                        weights.append(weight)
                                
                                # Normalize weights
                                if sum(weights) == 0:
                                    st.warning("All weights set to 0 - using equal weights instead")
                                    weights = np.ones(len(weights)) / len(weights)
                                else:
                                    weights = np.array(weights) / sum(weights)
                            
                            # Display weights
                            weights_df = pd.DataFrame({
                                'Criterion': list(available_criteria.keys()),
                                'Weight': weights,
                                'Direction': ['Maximize' if d == 1 else 'Minimize' for d in available_criteria.values()]
                            }).sort_values('Weight', ascending=False)
                            
                            with st.expander("View Weights"):
                                st.dataframe(
                                    weights_df.style.format({'Weight': '{:.2%}'}),
                                    use_container_width=True
                                )
                            
                            # Run analysis
                            if st.button("🚀 Run MCDM Analysis", type="primary", key="run_mcdm_custom"):
                                with st.spinner("Performing analysis..."):
                                    matrix = filtered_df_2[list(available_criteria.keys())].values
                                    types = np.array([available_criteria[k] for k in available_criteria])
                                    
                                    if mcdm_method == "TOPSIS":
                                        scores = run_topsis(matrix, weights, types)
                                        ranks = rankdata(scores, reverse=True).astype(int)
                                        results = pd.DataFrame({
                                            'Material': filtered_df_2['Name'].values,
                                            'Score': scores,
                                            'Rank': ranks
                                        }).sort_values('Rank')
                                    else:
                                        flows = run_promethee(matrix, weights, types)
                                        ranks = rankdata(flows, reverse=True).astype(int)
                                        results = pd.DataFrame({
                                            'Material': filtered_df_2['Name'].values,
                                            'Net Flow': flows,
                                            'Rank': ranks
                                        }).sort_values('Rank')
                                
                                # Display results
                                st.subheader("📋 MCDM Results")
                                st.dataframe(
                                    results.style.format({
                                        'Score': '{:.2f}',
                                        'Net Flow': '{:.2f}',
                                        'Rank': '{:.0f}'
                                    }),
                                    use_container_width=True
                                )
                                
                                # Visualize top materials
                                st.subheader("🏆 Top Materials")
                                top_n = min(3, len(results))
                                top_materials = results.head(top_n)['Material'].tolist()
                                
                                cols_top = st.columns(top_n)
                                for i, material in enumerate(top_materials):
                                    with cols_top[i]:
                                        st.metric(
                                            label=f"Rank #{int(results.iloc[i]['Rank'])}",
                                            value=material,
                                            help=f"Score: {results.iloc[i]['Score'] if 'Score' in results.columns else results.iloc[i]['Net Flow']:.4f}"
                                        )
                                
                                # Download results
                                excel_data = create_full_output(filtered_df_2, results, weights_df)
                                st.download_button(
                                    label="📥 Download Full MCDM Report",
                                    data=excel_data,
                                    file_name=f"custom_analysis_mcdm_{mcdm_method}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_mcdm_custom"
                                )
                        else:
                            st.warning("No materials match all the filters. Please adjust your criteria.")
                
                # Data table
                with st.expander("📋 View Data"):
                    st.dataframe(filtered_df[[x_col, y_col, "Name"]].sort_values(y_col, ascending=False))
                
                # Download
                st.download_button(
                    label="📥 Download Analysis Data",
                    data=filtered_df.to_csv(index=False).encode('utf-8'),
                    file_name="custom_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No materials match the current filters. Please adjust your criteria.")

if __name__ == "__main__":
    main()



