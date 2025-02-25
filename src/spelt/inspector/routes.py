from flask import Blueprint, render_template, current_app, request, jsonify, session
from typing import Optional, Dict, List, Union, Tuple, Any, TypeVar, overload, Callable, Set
from pathlib import Path
import json
import numpy as np
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.models import (
    ColumnDataSource, HoverTool, CrosshairTool, CustomJSHover,
    LinearAxis, Range1d, Slider, Column, CustomJS,
    PanTool, WheelZoomTool, BoxZoomTool, ResetTool, SaveTool, Button, LinearColorMapper, ColorBar
)
from bokeh.layouts import column
from bokeh.palettes import Category10
import re
from spelt.signal import Signal, MetadataKeys, SignalDomain, SamplingType
from enum import Enum
from .js_snippets import FORMAT_WITH_PREFIX, ZOOM_KEY_HANDLER, SLIDER_UPDATE_CALLBACK
import os

bp = Blueprint('main', __name__)

def serialize_signal(signal: Signal, representation: str, plot_type: str = 'line') -> dict:
    """Convert Signal to JSON serializable format."""
    serialized = {}
    serialized['path'] = str(signal.path.relative_to(Path(session['ROOT_DIR'])))
    serialized['shape'] = signal.shape  # Add shape information

    # First copy all metadata
    for key, value in signal.metadata.items():
        if isinstance(value, Enum):
            serialized[key] = value.value
        elif isinstance(value, (list, tuple)):
            serialized[key] = [item.value if isinstance(item, Enum) else item for item in value]
        else:
            serialized[key] = value

    # For heatmap, preserve 2D structure
    if plot_type == 'heatmap' and signal.ndim >= 2:
        serialized['data_2d'] = True
        return serialized

    # Handle name for complex signals
    if signal.data.dtype.kind == 'c':
        base_name = signal.metadata.get(MetadataKeys.NAME.value, '')
        rep_prefix = {
            'real': 'real',
            'imaginary': 'imag',
            'magnitude': 'mag',
            'phase': 'phase'
        }.get(representation, 'real')
        serialized[MetadataKeys.NAME.value] = f"{rep_prefix}({base_name})"

    serialized['representation'] = representation
    return serialized

def create_new_tab():
    """Create a new tab"""
    # Find a unique ID for the new tab
    tab_id = 1
    while f'{tab_id}' in session['tabs']:
        tab_id += 1
    reset_tab(str(tab_id))
    session['n_tabs'] += 1
    return tab_id

def get_tab(tab_id):
    tab_id = str(tab_id)
    if not tab_id in session['tabs']:
        reset_tab(tab_id)
    return session['tabs'][tab_id]

def reset_tab(tab_id):
    tab_id = str(tab_id)
    session['tabs'][tab_id] = {
        'sub_plots': {},
        'n_plots': 0,
        'plot_type': 'line',  # Default to line plot
        'plot_style': 'line',  # Default plot style
        'x_scale': 'linear'   # Add x_scale at tab level
    }
    session.modified = True
    return session['tabs'][tab_id]

def delete_tab(tab_id):
    """Delete a tab"""
    tab_id = str(tab_id)
    del session['tabs'][tab_id]
    session['n_tabs'] -= 1
    session.modified = True

def add_new_plot(tab_state):
    """Adds a new sub plot to the tab"""
    # Find a unique ID for the new sub plot
    plot_id = tab_state['n_plots']
    tab_state['n_plots'] += 1
    tab_state['sub_plots'][f'plot_{plot_id}'] = {
        'signals': [],
        'position': plot_id, # New plots are added to the end
        'y_scale': 'linear'   # Only y_scale remains at plot level
    }
    session.modified = True
    return tab_state['sub_plots'][f'plot_{plot_id}']

def get_plot_state(tab_state, plot_id):
    return tab_state['sub_plots'][f'plot_{plot_id}']

def add_signal_to_plot(plot_state, signal: Signal, representation: str, plot_type: str = 'line', selected_column: Optional[int] = None):
    """Adds a signal to the plot."""
    # Check if signal is already in the plot with same representation and column
    signal_path = str(signal.path)
    for signal_dict in plot_state['signals']:
        if signal_dict['path'] in signal_path and signal_dict.get('representation') == representation:
            # For 2D signals, also check the column
            if signal.ndim == 2 and selected_column != 'all':
                # Check if this exact column is already plotted
                index_tuple = signal_dict.get('index_tuple', [])
                if len(index_tuple) > 1 and index_tuple[1] == selected_column:
                    return plot_state
            else:
                # For non-2D signals or when plotting all columns, use original behavior
                return plot_state

    if plot_type == 'heatmap':
        if signal.ndim == 2:
            # For heatmap, add the signal directly with 2D data
            plot_state['signals'].append(serialize_signal(signal, representation, plot_type='heatmap'))
        elif signal.ndim == 3:
            # Initialize slider state if not exists
            if 'sliders' not in plot_state:
                plot_state['sliders'] = {}
            # Add slider for third dimension
            plot_state['sliders'][signal_path] = {
                'current_index': 0,
                'dimension_size': signal.shape[2],
                'dimension_name': signal.dimensions[2] if len(signal.dimensions) > 2 else 'dim_2'
            }
            signal_2d = signal[:, :, 0] # Initially first index
            serialized = serialize_signal(signal_2d, representation, plot_type='heatmap')
            serialized['index'] = 0
            plot_state['signals'].append(serialized)
    elif plot_type == 'line' or plot_type == 'histogram':
        if signal.ndim == 1:
            # For 1D signals, add directly
            plot_state['signals'].append(serialize_signal(signal, representation, plot_type='line'))
        elif signal.ndim == 2:
            # For 2D signals in line mode, convert to list of 1D signals
            if selected_column is not None and selected_column != 'all':
                # Extract only the selected column
                selected_idx = int(selected_column)
                # Create new metadata for 1D signal
                new_metadata = {
                    MetadataKeys.UNIT.value: signal.metadata.get(MetadataKeys.UNIT.value),
                    MetadataKeys.NAME.value: f"{signal.metadata.get(MetadataKeys.NAME.value, '')} {selected_idx}",  # Add column index to name
                    MetadataKeys.X_UNIT.value: signal.metadata.get(MetadataKeys.X_UNIT.value),
                    MetadataKeys.X_NAME.value: signal.metadata.get(MetadataKeys.X_NAME.value),
                    MetadataKeys.DOMAIN.value: signal.metadata.get(MetadataKeys.DOMAIN.value),
                    MetadataKeys.SAMPLING_TYPE.value: signal.metadata.get(MetadataKeys.SAMPLING_TYPE.value),
                    MetadataKeys.DIMENSIONS.value: [signal.dimensions[0]]  # Only keep first dimension
                }
                # Copy dimension units if they exist
                if MetadataKeys.DIM_UNITS.value in signal.metadata:
                    new_metadata[MetadataKeys.DIM_UNITS.value] = [signal.metadata[MetadataKeys.DIM_UNITS.value][0]]

                signal_1d = signal.derive(signal.data[:, selected_idx], new_metadata)
                serialized = serialize_signal(signal_1d, representation, plot_type='line')
                serialized['index_tuple'] = [serialize_index(slice(None)), serialize_index(selected_idx)]
                plot_state['signals'].append(serialized)
            else:
                # Original behavior for all columns
                signals_1d, index_tuple_list = signal.to_1d_signals(slice(None))
                for sig_1d, index_tuple in zip(signals_1d, index_tuple_list):
                    serialized = serialize_signal(sig_1d, representation, plot_type='line')
                    serialized['index_tuple'] = [serialize_index(idx) for idx in index_tuple]
                    plot_state['signals'].append(serialized)
        elif signal.ndim == 3:
            # Initialize slider state if not exists
            if 'sliders' not in plot_state:
                plot_state['sliders'] = {}

            # Add slider for third dimension
            plot_state['sliders'][signal_path] = {
                'current_index': 0,
                'dimension_size': signal.shape[2],
                'dimension_name': signal.dimensions[2] if len(signal.dimensions) > 2 else 'dim_2'
            }

            # Convert to 1D signals using current slider value (0 initially)
            signals_1d, index_tuple_list = signal.to_1d_signals(slice(None), 0)
            for sig_1d, index_tuple in zip(signals_1d, index_tuple_list):
                serialized = serialize_signal(sig_1d, representation, plot_type='line')
                serialized['index_tuple'] = [serialize_index(idx) for idx in index_tuple]
                plot_state['signals'].append(serialized)

    session.modified = True
    return plot_state

def calculate_plot_height(n_plots):
    """Calculate plot height based on number of plots"""
    min_height = 150
    max_height = 600
    delta_height = 150
    height = max_height - (n_plots - 1) * delta_height
    return max(height, min_height)

def get_signals(tab_id, plot_id=None, return_as_dict=False):
    """Get all signals in the tab"""
    signals = []
    for plot_id_0, plot_state in get_tab(tab_id)['sub_plots'].items():
        if plot_id_0 == plot_id or plot_id is None:
            for signal_dict in plot_state['signals']:
                if return_as_dict:
                    signals.append(signal_dict)
                else:
                    signal_path = signal_dict['path']
                    signal = Signal.load(Path(session['ROOT_DIR']) / signal_path)
                    if 'index_tuple' in signal_dict:
                        index_tuple = [deserialize_index(idx) for idx in signal_dict['index_tuple']]
                        signal = signal[index_tuple]
                    elif signal_dict.get('data_2d', False) and 'index' in signal_dict:
                        signal = signal[:, :, signal_dict['index']]
                    if 'representation' in signal_dict:
                        # Handle different signal representations
                        representation = signal_dict['representation']
                        if representation == 'real':
                            signal.data = np.real(signal.data)
                        elif representation == 'imaginary':
                            signal.data = np.imag(signal.data)
                        elif representation == 'magnitude':
                            signal.data = np.abs(signal.data)
                        elif representation == 'phase':
                            signal.data = np.angle(signal.data)
                    if 'name' in signal_dict:
                        signal = signal.set_metadata('name', signal_dict['name'])

                    signals.append(signal)
    return signals



def find_x_range(signals: List[Signal], x_scale='linear'):
    """Calculate x range from all signals"""
    x_min = float('inf')  # Initialize to positive infinity
    x_max = float('-inf')  # Initialize to negative infinity
    for signal in signals:
        if signal.x_data is not None:
            if x_scale == 'linear':
                x_min = min(x_min, np.min(signal.x_data))
                x_max = max(x_max, np.max(signal.x_data))
            else:
                x_max = max(x_max, np.max(signal.x_data))
                # If np.min(signal.x_data) is 0, use the second smallest value as x_min
                x_min = min(x_min, np.min(signal.x_data[signal.x_data != 0]))

    # Handle case where no valid signals were found
    if x_min == float('inf'):
        x_min = 0
        x_max = max(len(signal.data) for signal in signals)

    x_start = x_min - (x_max - x_min) * 0.05 if x_scale == 'linear' else x_min
    x_end = x_max + (x_max - x_min) * 0.05
    return Range1d(start=x_start, end=x_end)

def get_tools():
    """Get tools for the plot"""
    # Create shared tools
    pan = PanTool()
    wheel_zoom_x = WheelZoomTool(dimensions="width")
    wheel_zoom_y = WheelZoomTool(dimensions="height")
    box_zoom = BoxZoomTool()
    reset = ResetTool()
    save = SaveTool()
    crosshair = CrosshairTool(
        line_color='gray',
        line_alpha=0.5,
        line_width=1
    )

    tools = [box_zoom, pan, wheel_zoom_x, wheel_zoom_y, reset, save, crosshair]
    return tools

@bp.route('/')
def index():
    """Render main page."""
    # Initialize session state
    session['tabs'] = {}
    session['n_tabs'] = 1
    if not 'ROOT_DIR' in session:
        session['ROOT_DIR'] = current_app.config.get('DEFAULT_ROOT_DIR', str(Path.cwd()))
    reset_tab('1')
    return render_template('index.html')

@bp.route('/browse/')
@bp.route('/browse/<path:subpath>')
def browse(subpath=''):
    """Browse directory contents."""
    try:
        # Get absolute path of root directory
        root_dir = Path(session['ROOT_DIR']).resolve()

        # Get full path and ensure it exists
        full_path = (root_dir / subpath).resolve()
        if not full_path.exists():
            return jsonify({'error': 'Directory not found'}), 404

        # Get parent path relative to root
        parent_path = None
        if subpath:
            try:
                parent_path = str(full_path.parent.relative_to(root_dir))
            except ValueError:
                parent_path = None

        # List directory contents
        items = []
        for p in full_path.iterdir():
            if p.name.startswith('.'):
                continue
            if p.is_dir():
                # Check if directory contains signals or has signal metadata
                has_signals = (p / '.signals.json').exists()
                is_signal = (p / '.signal').exists()

                if has_signals or is_signal:
                    try:
                        # Use absolute paths for signal loading
                        item_data = {
                            'name': p.name,
                            'type': 'signal' if is_signal else 'folder',
                            'path': str(Path(os.path.relpath(p, root_dir))),
                        }

                        # If it's a signal, load it to get properties
                        if is_signal:
                            signal = Signal.load(p)  # Use absolute path
                            item_data.update({
                                'ndim': signal.ndim,
                                'shape': signal.shape,  # Add shape information
                                'is_complex': signal.data.dtype.kind == 'c',
                                'description': signal.metadata.get(MetadataKeys.DESCRIPTION.value, '')
                            })
                        items.append(item_data)
                    except Exception as e:
                        import traceback
                        print(f"Error processing path {p}: {e}")
                        print("Traceback:")
                        print(traceback.format_exc())
                        continue

        # Sort items alphabetically by name, case-insensitive
        items.sort(key=lambda x: x['name'].lower())

        return jsonify({
            'items': items,
            'current_path': str(full_path),
            'parent_path': parent_path,
            'is_root': not bool(subpath)
        })

    except Exception as e:
        print(f"Error browsing directory: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/plot/update')
def update_plot():
    tab_id = request.args.get('tab_id', '1')

    # Check if session is lost
    if 'tabs' not in session or tab_id not in session['tabs']:
        return jsonify({
            'status': 'error',
            'error': 'session_lost',
            'message': 'Session data was lost. Please reload the page.'
        }), 400

    tab_state = get_tab(tab_id)
    plot_type = tab_state.get('plot_type', 'line')

    if plot_type == 'heatmap':
        return update_heatmap_plot(tab_id)
    elif plot_type == 'histogram':
        return update_histogram_plot(tab_id)
    else:
        return update_line_plot(tab_id)

def update_heatmap_plot(tab_id):
    """Create heatmap plot for 2D/3D signals."""
    tab_state = get_tab(tab_id)
    plot_state = get_plot_state(tab_state, 0)
    signals = get_signals(tab_id)
    # Get the first signal (heatmap mode only supports one signal)
    if not signals:
        return jsonify({'error': 'No signals to plot'}), 400

    # Create figure
    p = figure(
        height=600,
        sizing_mode='stretch_width',
        toolbar_location='above',
        tools=get_tools(),
        tooltips=[
            ('x', '$x{0}'),
            ('y', '$y{0}'),  # Keep original y coordinate for tooltip
            ('value', '@image{0.000}')
        ]
    )

    # Configure axes
    p.grid.grid_line_color = None

    signal = signals[0]
    # Get data and dimensions
    data = np.flipud(signal.data)  # Flip the data array vertically

    # Get color range from tab state
    color_range = tab_state.get('color_range', {})
    color_min = color_range.get('min', np.min(data))
    color_max = color_range.get('max', np.max(data))

    # Create color mapper with custom range
    mapper = LinearColorMapper(palette="Viridis256", low=color_min, high=color_max)

    # Create heatmap with hover tool
    p.image(
        image=[data],
        x=0,
        y=0,  # Start from bottom
        dw=data.shape[1],
        dh=data.shape[0],  # Keep height positive
        color_mapper=mapper,
        name='heatmap'  # Add name for hover tool reference
    )

    # Remove the y-axis range flip since we're flipping the data instead
    p.y_range.start = 0  # Start from bottom
    p.y_range.end = data.shape[0]  # End at height

    # Add colorbar
    color_bar = ColorBar(
        color_mapper=mapper,
        label_standoff=12,
        border_line_color=None,
        location=(0, 0)
    )
    p.add_layout(color_bar, 'right')

    # Configure axes with dim units if applicable
    dim_units = signal.dim_units
    if len(dim_units) == 2:
        p.xaxis.axis_label = dim_units[0]
        p.yaxis.axis_label = dim_units[1]

    # Set plot title to sigal name if available
    if signal.metadata.get(MetadataKeys.NAME.value, None):
        p.title.text = signal.metadata.get(MetadataKeys.NAME.value, '')

    # Add sliders for 3D signals
    sliders = []
    if 'sliders' in plot_state:
        for signal_path, slider_state in plot_state['sliders'].items():
            # Add slider for third dimension
            slider = Slider(
                    start=0,
                    end=slider_state['dimension_size'] - 1,
                    value=slider_state['current_index'],
                    step=1,
                    title=f"{slider_state['dimension_name']} Index",
                    sizing_mode='stretch_width'
                )

            # Create callback to update plot when slider changes
            callback = CustomJS(
                    args=dict(
                        signal_path=str(Path(signal_path).relative_to(Path(session['ROOT_DIR']))),
                        plot_id='plot_0',
                        tab_id=tab_id,
                        representation=plot_state['signals'][0].get('representation', 'real')
                    ),
                    code="""
                    // Send the new slice index to the server
                    fetch(`/signal/update_slice/${signal_path}?tab_id=${tab_id}&plot_id=${plot_id}&slice_index=${cb_obj.value}&representation=${representation}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                // Trigger plot update
                                window.updatePlot(tab_id);
                            }
                        });
                    """
                )
            slider.js_on_change('value', callback)
            sliders.append(slider)

    if sliders:
        layout = column(
            *sliders,
            p,
            sizing_mode='stretch_width',
            styles={'width': '100%'}
        )
    else:
        layout = p

    script, div = components(layout)

    # Extract JavaScript
    script_content = re.search(r'<script.*?>(.*?)</script>', script, re.DOTALL)
    if script_content:
        script = script_content.group(1)

    return jsonify({
        'script': script,
        'div': div,
        'status': 'success'
    })

def update_line_plot(tab_id):
    """Create line plot for 1D signals or 2D/3D signals in line mode."""
    signals = get_signals(tab_id)
    # verify that all signals have the same x_unit
    x_units = [signal.metadata.get(MetadataKeys.X_UNIT.value, None) for signal in signals]
    if not all(x_unit == x_units[0] for x_unit in x_units):
        return jsonify({'error': 'All signals must have the same x_unit'}), 400

    tab_state = get_tab(tab_id)
    # Get scale settings - x_scale from tab_state, y_scale from plot_state
    x_scale = tab_state.get('x_scale', 'linear')

    x_range = find_x_range(signals, x_scale=x_scale)

    # Get color palette
    colors = Category10[10]

    # Preallocate empty list for plots, using n_plots from tab state
    n_plots = tab_state['n_plots']
    plots = [None] * n_plots
    sliders = []
    # Create base tools without hover
    tools = get_tools()

    for plot_id, plot_state in tab_state['sub_plots'].items():
        position = plot_state['position']

        # Group signals by unit first
        signals_in_plot = get_signals(tab_id, plot_id)
        signals_by_unit = {}
        for signal in signals_in_plot:
            unit = signal.metadata.get(MetadataKeys.UNIT.value, None)
            if unit not in signals_by_unit:
                signals_by_unit[unit] = []
            signals_by_unit[unit].append(signal)

        # only show legends and hover if less than 10 signals in plot
        SHOW_LEGENDS = len(signals_in_plot) <= 15

        y_scale = plot_state.get('y_scale', 'linear')
        p = figure(
            height=calculate_plot_height(n_plots),  # Smaller height for sub plots
            sizing_mode='stretch_width',
            x_range=x_range,  # Use shared x range
            tools=tools,
            toolbar_location='above' if position == 0 else None,
            x_axis_type=x_scale,
            y_axis_type=y_scale,
            active_drag=tools[0]
        )

        # For log scale, ensure positive ranges
        print(x_range.start)
        if x_scale == 'log':
            if isinstance(x_range, Range1d):
                x_range.start = max(1e-10, x_range.start)
        if y_scale == 'log':
            p.y_range.start = max(1e-10, p.y_range.start or 1e-10)

        # Initialize extra y ranges dictionary
        p.extra_y_ranges = {}

        # Add grid
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.3
        p.grid.minor_grid_line_color = 'gray'
        p.grid.minor_grid_line_alpha = 0.1

        # Configure axes
        if position == n_plots - 1:
            p.xaxis.axis_label = signals[0].metadata.get(MetadataKeys.X_NAME.value, '')
        else:
            p.xaxis.visible = False
            p.xgrid.visible = False

        # Create y-axes and plot signals
        color_index = 0
        for i, (unit, unit_signals) in enumerate(signals_by_unit.items()):
            # Create y axis
            if i == 0:
                # Use primary y axis for first unit
                y_axis = p.yaxis[0]
                y_axis.axis_label = unit if unit else 'Value'
            else:
                # Create new y axis for additional units
                y_axis = LinearAxis(
                    y_range_name=f'y{i}',
                    axis_label=unit if unit else 'Value'
                )
                p.add_layout(y_axis, 'right')
                # Add new range
                p.extra_y_ranges[f'y{i}'] = Range1d(start=0, end=1)  # Will be auto-scaled

            # Get plot style from plot state
            plot_style = plot_state.get('plot_style', 'line')

            # Plot signals for this unit
            for signal in unit_signals:
                y_data = signal.data
                # Remove inf values from data
                valid_indices = ~np.isinf(y_data)
                y_data = y_data[valid_indices]
                x_data = signal.x_data if signal.x_data is not None else np.arange(y_data.shape[0])
                x_data = x_data[valid_indices]
                legend = signal.metadata.get(MetadataKeys.NAME.value, f'Signal {color_index+1}')
                color = colors[color_index % len(colors)]

                # Create data source with correct y field name
                source_data = {'x': x_data, 'y': y_data}
                source = ColumnDataSource(source_data)

                # Create hover tool specific to this line
                if SHOW_LEGENDS:
                    hover = HoverTool(
                            tooltips=[
                                ('x', '@x{custom}'),
                            (unit if unit else 'Value', '@y{custom}')
                        ],
                        formatters={
                            '@x': CustomJSHover(code=FORMAT_WITH_PREFIX),
                            '@y': CustomJSHover(code=FORMAT_WITH_PREFIX)
                        },
                        mode='vline',
                        renderers=[] # Will be set after line is created
                    )

                    # Plot with appropriate style and y range
                    if plot_style == 'scatter':
                        if i == 0:
                            line = p.scatter('x', 'y', source=source, size=4, color=color,
                                        legend_label=legend)
                        else:
                            line = p.scatter('x', 'y', source=source, size=4, color=color,
                                        legend_label=legend, y_range_name=f'y{i}')
                    else:  # line style
                        if i == 0:
                            line = p.line('x', 'y', source=source, line_width=2, color=color,
                                        legend_label=legend)
                        else:
                            line = p.line('x', 'y', source=source, line_width=2, color=color,
                                        legend_label=legend, y_range_name=f'y{i}')

                    # Add line to hover tool's renderers
                    hover.renderers = [line]
                    p.add_tools(hover)
                else:
                    if plot_style == 'scatter':
                        line = p.scatter('x', 'y', source=source, size=4, color=color)
                    else:  # line style
                        line = p.line('x', 'y', source=source, line_width=2, color=color)

                color_index += 1

            # Auto-scale y range
            y_min = float('inf')
            y_max = float('-inf')
            for signal in unit_signals:
                # Remove inf values from data
                y_data = np.real(signal.data)
                valid_indices = ~np.isinf(y_data)
                y_data = y_data[valid_indices]
                y_min = min(y_min, np.min(y_data))
                y_max = max(y_max, np.max(y_data))

            margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            if i == 0:
                p.y_range.start = y_min - margin
                p.y_range.end = y_max + margin
            else:
                p.extra_y_ranges[f'y{i}'].start = y_min - margin
                p.extra_y_ranges[f'y{i}'].end = y_max + margin

        # Add sliders for 3D signals
        if 'sliders' in plot_state:
            for signal_path, slider_state in plot_state['sliders'].items():
                slider = Slider(
                    start=0,
                    end=slider_state['dimension_size'] - 1,
                    value=slider_state['current_index'],
                    step=1,
                    title=f"{slider_state['dimension_name']} Index",
                    sizing_mode='stretch_width'
                )

                # Create callback to update plot when slider changes
                callback = CustomJS(
                    args=dict(
                        signal_path=str(Path(signal_path).relative_to(Path(session['ROOT_DIR']))),
                    plot_id=plot_id,
                    tab_id=tab_id,
                    representation=plot_state['signals'][0].get('representation', 'real')
                ),
                code="""
                // Send the new slice index to the server
                fetch(`/signal/update_slice/${signal_path}?tab_id=${tab_id}&plot_id=${plot_id}&slice_index=${cb_obj.value}&representation=${representation}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            // Trigger plot update
                            window.updatePlot(tab_id);
                        }
                    });
                """
            )
            slider.js_on_change('value', callback)
            sliders.append(slider)

        if SHOW_LEGENDS:
            p.legend.click_policy = "hide"
            p.legend.location = "top_right"
            p.legend.background_fill_alpha = 0.7

        plots[position] = p

    # Create layout with sliders if any
    if sliders:
        layout = column(
            *sliders,
            *plots,
            sizing_mode='stretch_width',
            styles={'width': '100%'}
        )
    else:
        layout = column(
            *plots,
            sizing_mode='stretch_width',
            styles={'width': '100%'}
        )

    script, div = components(layout)

    # Extract JavaScript
    script_content = re.search(r'<script.*?>(.*?)</script>', script, re.DOTALL)
    if script_content:
        script = script_content.group(1)

    return jsonify({
        'script': script,
        'div': div,
        'status': 'success'
    })


def update_histogram_plot(tab_id):
    """Create histogram plot for signals."""
    tab_state = get_tab(tab_id)
    signals = get_signals(tab_id)

    # Create figure
    p = figure(
        height=600,
        sizing_mode='stretch_width',
        toolbar_location='above',
        tools=get_tools(),
        x_axis_label='Value',
        y_axis_label='Count'
    )

    # Get color palette
    colors = Category10[10]
    color_index = 0

    # Process each signal
    for signal in signals:
        # Get the data based on signal dimensionality
        if signal.ndim == 1:
            # For 1D signals, use data directly
            data_to_plot = [signal.data]
            labels = [signal.metadata.get(MetadataKeys.NAME.value, f'Signal {color_index+1}')]

        elif signal.ndim == 2:
            # For 2D signals, separate each column
            data_to_plot = [signal.data[:, i] for i in range(signal.shape[1])]
            labels = [f"{signal.metadata.get(MetadataKeys.NAME.value, 'Signal')} {i}"
                     for i in range(signal.shape[1])]

        elif signal.ndim == 3:
            # For 3D signals, use current slice index from slider
            slider_state = find_slider_state(tab_state['sub_plots']['plot_0'], str(signal.path))
            if slider_state:
                slice_idx = slider_state['current_index']
                data_to_plot = [signal.data[:, i, slice_idx] for i in range(signal.shape[1])]
                labels = [f"{signal.metadata.get(MetadataKeys.NAME.value, 'Signal')} {i}"
                         for i in range(signal.shape[1])]
            else:
                continue  # Skip if no slider state found

        # Create histogram for each data array
        for data_array, label in zip(data_to_plot, labels):
            # Remove any inf or nan values
            valid_data = data_array[~np.isinf(data_array) & ~np.isnan(data_array)]
            if len(valid_data) == 0:
                continue

            # Calculate histogram
            hist, edges = np.histogram(valid_data, bins=50, density=False)

            # Create data source
            source = ColumnDataSource({
                'top': hist,
                'left': edges[:-1],
                'right': edges[1:],
                'name': [label] * len(hist)
            })

            # Add histogram bars
            p.quad(
                top='top',
                bottom=0,
                left='left',
                right='right',
                source=source,
                fill_color=colors[color_index % len(colors)],
                line_color='white',
                alpha=0.7,
                legend_label=label
            )
            color_index += 1

    # Configure hover tool
    hover = HoverTool(
        tooltips=[
            ('Range', '(@left{0.00}, @right{0.00})'),
            ('Count', '@top'),
            ('Name', '@name')
        ]
    )
    p.add_tools(hover)

    # Configure legend
    if color_index > 0:  # Only add legend if we have plots
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"

    # Add sliders for 3D signals
    sliders = []
    for plot_id, plot_state in tab_state['sub_plots'].items():
        if 'sliders' in plot_state:
            for signal_path, slider_state in plot_state['sliders'].items():
                slider = Slider(
                    start=0,
                    end=slider_state['dimension_size'] - 1,
                    value=slider_state['current_index'],
                    step=1,
                    title=f"{slider_state['dimension_name']} Index",
                    sizing_mode='stretch_width'
                )

                callback = CustomJS(
                    args=dict(
                        signal_path=signal_path,
                        plot_id=plot_id,
                        tab_id=tab_id,
                        representation=plot_state['signals'][0].get('representation', 'real')
                    ),
                    code="""
                    fetch(`/signal/update_slice/${signal_path}?tab_id=${tab_id}&plot_id=${plot_id}&slice_index=${cb_obj.value}&representation=${representation}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                window.updatePlot(tab_id);
                            }
                        });
                    """
                )
                slider.js_on_change('value', callback)
                sliders.append(slider)

    if sliders:
        layout = column(
            *sliders,
            p,
            sizing_mode='stretch_width',
            styles={'width': '100%'}
        )
    else:
        layout = p

    script, div = components(layout)

    # Extract JavaScript
    script_content = re.search(r'<script.*?>(.*?)</script>', script, re.DOTALL)
    if script_content:
        script = script_content.group(1)

    return jsonify({
        'script': script,
        'div': div,
        'status': 'success'
    })


@bp.route('/signal/replace/<path:signal_path>')
def replace(signal_path):
    tab_id = request.args.get('tab_id', '1')
    representation = request.args.get('representation', 'real')
    plot_type = request.args.get('plot_type', 'line')
    selected_column = request.args.get('selected_column', 'all')

    # Get full path
    full_path = Path(session['ROOT_DIR']) / signal_path
    if not (full_path / '.signal').exists():
        return jsonify({'error': 'Not a signal directory'}), 400

    # Load signal
    signal = Signal.load(full_path)
    tab_state = reset_tab(tab_id)
    tab_state['plot_type'] = plot_type  # Set plot type for the tab
    session.modified = True

    plot_state = add_new_plot(tab_state)
    add_signal_to_plot(plot_state, signal, representation, plot_type, selected_column)

    return jsonify({'status': 'success'})


@bp.route('/signal/append/<path:signal_path>')
def append(signal_path):
    tab_id = request.args.get('tab_id', '1')
    representation = request.args.get('representation', 'real')
    selected_column = request.args.get('selected_column', 'all')

    # Get tab state and check plot type
    tab_state = get_tab(tab_id)
    if tab_state['plot_type'] == 'heatmap':
        return jsonify({'error': 'Cannot append to heatmap plot'}), 400

    # Get full path
    full_path = Path(session['ROOT_DIR']) / signal_path
    if not (full_path / '.signal').exists():
        return jsonify({'error': 'Not a signal directory'}), 400

    # Load signal
    signal = Signal.load(full_path)
    # Add new plot if none exist, otherwise use last plot
    if tab_state['n_plots'] == 0:
        plot_state = add_new_plot(tab_state)
    else:
        plot_state = get_plot_state(tab_state, tab_state['n_plots'] - 1)
    add_signal_to_plot(plot_state, signal, representation, 'line', selected_column)

    return jsonify({'status': 'success'})

@bp.route('/signal/new_tab/<path:signal_path>')
def new_tab(signal_path):
    # Plot signal in new tab
    representation = request.args.get('representation', 'real')
    plot_type = request.args.get('plot_type', 'line')
    selected_column = request.args.get('selected_column', 'all')
    tab_id = create_new_tab()

    # Get full path
    full_path = Path(session['ROOT_DIR']) / signal_path
    if not (full_path / '.signal').exists():
        return jsonify({'error': 'Not a signal directory'}), 400

    # Load signal
    signal = Signal.load(full_path)
    tab_state = get_tab(tab_id)
    tab_state['plot_type'] = plot_type  # Set plot type for the tab
    # By default, add signal to last plot
    plot_state = add_new_plot(tab_state)
    add_signal_to_plot(plot_state, signal, representation, plot_type, selected_column)

    return jsonify({'status': 'success', 'tab_id': tab_id})

@bp.route('/plot/close_tab/<int:tab_id>')
def close_tab(tab_id):
    tab_id = str(tab_id)

    # Do not delete tab is it is the only one
    if session['n_tabs'] == 1:
        return jsonify({'status': 'error', 'message': 'Cannot close the only tab'})

    # Close tab
    delete_tab(tab_id)
    return jsonify({'status': 'success', 'tab_id': tab_id})

@bp.route('/signal/update_slice/<path:signal_path>')
def update_slice(signal_path):
    """Update the slice index for a 3D signal"""
    tab_id = request.args.get('tab_id')
    plot_id = request.args.get('plot_id')
    representation = request.args.get('representation', 'real')
    # Extract plot_id as int if it starts with "plot_"
    if isinstance(plot_id, str) and plot_id.startswith('plot_'):
        plot_id = int(plot_id[5:])  # Remove "plot_" prefix and convert to int

    slice_index = int(request.args.get('slice_index', 0))

    if tab_id is None or plot_id is None:
        return jsonify({'error': 'Missing tab_id or plot_id'}), 400

    # Get the plot state
    tab_state = get_tab(tab_id)
    plot_state = get_plot_state(tab_state, plot_id)
    plot_type = tab_state.get('plot_type', 'line')

    # Update slider state
    slider_state = find_slider_state(plot_state, signal_path)
    if slider_state is not None:
        slider_state['current_index'] = slice_index

        # Load the signal
        full_path = Path(session['ROOT_DIR']) / signal_path
        signal = Signal.load(full_path)

        # Remove old signals for this path
        plot_state['signals'] = [s for s in plot_state['signals']
                               if s['path'] != signal_path]

        # Add new signals with updated slice
        if plot_type == 'heatmap':
            # For heatmap, add the signal directly with 2D data
            serialized = serialize_signal(signal, representation, plot_type='heatmap')
            serialized['index'] = slice_index
            plot_state['signals'].append(serialized)
        else:
            # For line plot, convert to 1D signals
            signals_1d, index_tuple_list = signal.to_1d_signals(slice(None), slice_index)
            for sig_1d, index_tuple in zip(signals_1d, index_tuple_list):
                serialized = serialize_signal(sig_1d, representation, plot_type='line')
                serialized['index_tuple'] = [serialize_index(idx) for idx in index_tuple]
                plot_state['signals'].append(serialized)

        session.modified = True
        return jsonify({'status': 'success'})

    return jsonify({'error': 'Signal not found or not 3D'}), 404


@bp.route('/plot/split')
def split_plot():
    tab_id = request.args.get('tab_id')
    # Get list of all signals in tab
    tab_state = get_tab(tab_id)
    signal_dict_list = get_signals(tab_id, return_as_dict=True)
    tab_state = reset_tab(tab_id)

    # For each signal in signal_list, add a new plot and add signal to new plot
    for signal_dict in signal_dict_list:
        plot_state = add_new_plot(tab_state)
        plot_state['signals'].append(signal_dict)

    session.modified = True
    return jsonify({'status': 'success'})


@bp.route('/plot/combine')
def combine_plot():
    tab_id = request.args.get('tab_id')
    # Get list of all signals in tab
    tab_state = get_tab(tab_id)
    signal_dict_list = get_signals(tab_id, return_as_dict=True)
    tab_state = reset_tab(tab_id)
    # For each signal in signal_list, add a new plot and add signal to new plot
    plot_state = add_new_plot(tab_state)
    for signal_dict in signal_dict_list:
        plot_state['signals'].append(signal_dict)

    session.modified = True
    return jsonify({'status': 'success'})




def find_slider_state(plot_state, signal_path):
    if not 'sliders' in plot_state:
        return None
    for signal_path_0, slider_state in plot_state['sliders'].items():
        if signal_path in signal_path_0:
            return slider_state
    return None


def serialize_index(idx):
    """Convert index to JSON serializable format."""
    if isinstance(idx, slice):
        return {
            'type': 'slice',
            'start': idx.start if idx.start is not None else None,
            'stop': idx.stop if idx.stop is not None else None,
            'step': idx.step if idx.step is not None else None
        }
    return int(idx)  # Convert numpy integers to Python integers

def deserialize_index(idx):
    """Convert serialized index back to proper format."""
    if isinstance(idx, dict) and idx.get('type') == 'slice':
        return slice(idx['start'], idx['stop'], idx['step'])
    return idx

@bp.route('/plot/set_scale/<int:tab_id>')
def set_scale(tab_id):
    """Set the scale for an axis in a plot."""
    axis = request.args.get('axis')
    scale = request.args.get('scale')

    if axis not in ['x', 'y']:
        return jsonify({'error': 'Invalid axis'}), 400
    if scale not in ['linear', 'log']:
        return jsonify({'error': 'Invalid scale'}), 400

    tab_state = get_tab(str(tab_id))

    if axis == 'x':
        # Set x_scale at tab level
        tab_state['x_scale'] = scale
    else:
        # Update y_scale for all plots in the tab
        for plot_state in tab_state['sub_plots'].values():
            plot_state['y_scale'] = scale

    session.modified = True
    return jsonify({'status': 'success'})

@bp.route('/plot/set_color_range/<int:tab_id>')
def set_color_range(tab_id):
    """Set the color range for heatmap."""
    min_val = request.args.get('min', type=float)
    max_val = request.args.get('max', type=float)

    tab_state = get_tab(str(tab_id))
    if 'color_range' not in tab_state:
        tab_state['color_range'] = {}

    if min_val is not None:
        tab_state['color_range']['min'] = min_val
    if max_val is not None:
        tab_state['color_range']['max'] = max_val

    session.modified = True
    return jsonify({'status': 'success'})

@bp.route('/plot/set_style/<int:tab_id>')
def set_style(tab_id):
    """Set the plot style for all plots in a tab."""
    style = request.args.get('style')

    if style not in ['line', 'scatter']:
        return jsonify({'error': 'Invalid style'}), 400

    tab_state = get_tab(str(tab_id))

    # Update style for all plots in the tab
    for plot_state in tab_state['sub_plots'].values():
        plot_state['plot_style'] = style

    session.modified = True
    return jsonify({'status': 'success'})

