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
    PanTool, WheelZoomTool, BoxZoomTool, ResetTool, SaveTool, Button
)
from bokeh.layouts import column
from bokeh.palettes import Category10
import re
from siva.signal import Signal, MetadataKeys, SignalDomain, SamplingType
from enum import Enum
from .js_snippets import FORMAT_WITH_PREFIX, ZOOM_KEY_HANDLER, SLIDER_UPDATE_CALLBACK

bp = Blueprint('main', __name__)

def serialize_signal(signal: Signal) -> dict:
    """Convert Signal to JSON serializable format."""
    serialized = {}
    serialized['path'] = str(signal.path.relative_to(current_app.config['ROOT_DIR']))
    for key, value in signal.metadata.items():
        if isinstance(value, Enum):
            serialized[key] = value.value
        elif isinstance(value, (list, tuple)):
            serialized[key] = [item.value if isinstance(item, Enum) else item for item in value]
        else:
            serialized[key] = value
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
        'n_plots': 0
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
        'position': plot_id # New plots are added to the end
    }
    session.modified = True
    return tab_state['sub_plots'][f'plot_{plot_id}']

def get_plot_state(tab_state, plot_id):
    return tab_state['sub_plots'][f'plot_{plot_id}']

def add_signal_to_plot(plot_state, signal: Signal):
    """Adds a signal to the plot.

    For 2D signals, converts them to multiple 1D signals.
    For 1D signals, adds them directly.
    For 3D signals, adds slider for third dimension and converts to multiple 1D signals.
    """
    # Check if signal is already in the plot
    signal_path = str(signal.path)
    for signal_dict in plot_state['signals']:
        if signal_dict['path'] in signal_path:
            return plot_state

    if signal.ndim == 1:
        # For 1D signals, add directly
        plot_state['signals'].append(serialize_signal(signal))
    elif signal.ndim == 2:
        # For 2D signals, convert to list of 1D signals
        signals_1d, index_tuple_list = signal.to_1d_signals(slice(None))
        for sig_1d, index_tuple in zip(signals_1d, index_tuple_list):
            serialized = serialize_signal(sig_1d)
            serialized['index_tuple'] = [serialize_index(idx) for idx in index_tuple]
            plot_state['signals'].append(serialized)
    elif signal.ndim == 3:
        # Initialize slider state if not exists
        if 'sliders' not in plot_state:
            plot_state['sliders'] = {}

        # Add slider for third dimension
        plot_state['sliders'][signal_path] = {
            'current_index': 0,
            'dimension_size': signal.shape[2],  # Size of third dimension
            'dimension_name': signal.dimensions[2] if len(signal.dimensions) > 2 else 'dim_2'
        }

        # Convert to 1D signals using current slider value (0 initially)
        signals_1d, index_tuple_list = signal.to_1d_signals(slice(None), 0)
        for sig_1d, index_tuple in zip(signals_1d, index_tuple_list):
            serialized = serialize_signal(sig_1d)
            serialized['index_tuple'] = [serialize_index(idx) for idx in index_tuple]
            plot_state['signals'].append(serialized)
    else:
        # For 4D+ signals, take first slice of extra dimensions
        slice_args = [slice(None), slice(None)] + [0] * (signal.ndim - 2)
        signals_1d, index_tuple_list = signal.to_1d_signals(*slice_args[1:])
        for sig_1d, index_tuple in zip(signals_1d, index_tuple_list):
            serialized = serialize_signal(sig_1d)
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
                    signal = Signal.load(current_app.config['ROOT_DIR'] / signal_path)
                    if 'index_tuple' in signal_dict:
                        index_tuple = [deserialize_index(idx) for idx in signal_dict['index_tuple']]
                        signal = signal[index_tuple]
                    if 'name' in signal_dict:
                        signal = signal.set_metadata('name', signal_dict['name'])
                    signals.append(signal)
    return signals



def find_x_range(signals: List[Signal]):
    """Calculate x range from all signals"""
    x_min = float('inf')  # Initialize to positive infinity
    x_max = float('-inf')  # Initialize to negative infinity
    for signal in signals:
        if signal.x_data is not None:
            x_min = min(x_min, np.min(signal.x_data))
            x_max = max(x_max, np.max(signal.x_data))

    # Handle case where no valid signals were found
    if x_min == float('inf'):
        return None
    return Range1d(start=x_min - (x_max - x_min) * 0.05,
                                   end=x_max + (x_max - x_min) * 0.05)

def get_tools(units_list=None):
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

    tools = [pan, wheel_zoom_x, wheel_zoom_y, box_zoom, reset, save, crosshair]
    return tools

@bp.route('/')
def index():
    """Render main page."""
    # Initialize tabs state if not exists
    session['tabs'] = {}
    session['n_tabs'] = 1
    reset_tab('1')
    return render_template('index.html')

@bp.route('/browse/')
@bp.route('/browse/<path:subpath>')
def browse(subpath=''):
    """Browse directory contents."""
    try:
        # Get full path and ensure it exists
        full_path = current_app.config['ROOT_DIR'] / subpath
        if not full_path.exists():
            return jsonify({'error': 'Directory not found'}), 404

        # Get parent path (relative to root)
        parent_path = None
        if subpath:
            rel_parent = Path(subpath).parent
            parent_path = str(rel_parent) if str(rel_parent) != '.' else ''

        # List directory contents
        items = []
        for p in full_path.iterdir():
            if p.is_dir():
                # Check if directory contains signals or has signal metadata
                has_signals = (p / '.signals.json').exists()
                is_signal = (p / '.signal').exists()

                if has_signals or is_signal:
                    items.append({
                        'name': p.name,
                        'type': 'signal' if is_signal else 'folder',
                        'path': str(p.relative_to(current_app.config['ROOT_DIR']))
                    })

        return jsonify({
            'items': items,
            'current_path': subpath or '',
            'parent_path': parent_path,
            'is_root': not bool(subpath)
        })

    except Exception as e:
        print(f"Error browsing directory: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/set-root', methods=['POST'])
def set_root():
    """Set root directory."""
    try:
        new_root = Path(request.json['path'])
        if not new_root.exists() or not new_root.is_dir():
            return jsonify({'error': 'Invalid directory'}), 400

        current_app.config['ROOT_DIR'] = new_root
        print(f"Set root directory to: {new_root}")  # For debugging
        return jsonify({'success': True, 'path': str(new_root)})

    except Exception as e:
        print(f"Error setting root: {e}")  # For debugging
        return jsonify({'error': str(e)}), 500

@bp.route('/plot/update')
def update_plot():
    tab_id = request.args.get('tab_id', '1')
    # Create new figure
    signals = get_signals(tab_id)
    x_range = find_x_range(signals)

    # Get color palette
    colors = Category10[10]

    # Preallocate empty list for plots, using n_plots from tab state
    tab_state = get_tab(tab_id)
    n_plots = tab_state['n_plots']
    plots = [None] * n_plots
    sliders = []

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

        # Create base tools without hover
        tools = get_tools()

        p = figure(
            height=calculate_plot_height(n_plots),  # Smaller height for sub plots
            sizing_mode='stretch_width',
            x_range=x_range,  # Use shared x range
            tools=tools,
            toolbar_location='above' if position == 0 else None,
        )

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

            # Plot signals for this unit
            for signal in unit_signals:
                y_data = signal.data
                x_data = signal.x_data if signal.x_data is not None else np.arange(y_data.shape[0])
                legend = signal.metadata.get(MetadataKeys.NAME.value, f'Signal {color_index+1}')
                color = colors[color_index % len(colors)]

                # Create data source with correct y field name
                source_data = {'x': x_data, 'y': np.real(y_data)}
                source = ColumnDataSource(source_data)

                # Create hover tool specific to this line
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

                # Plot line with appropriate y range
                if i == 0:
                    line = p.line('x', 'y', source=source, line_width=2, color=color,
                              legend_label=legend)
                else:
                    line = p.line('x', 'y', source=source, line_width=2, color=color,
                              legend_label=legend, y_range_name=f'y{i}')

                # Add line to hover tool's renderers
                hover.renderers = [line]
                p.add_tools(hover)

                color_index += 1

            # Auto-scale y range
            y_min = float('inf')
            y_max = float('-inf')
            for signal in unit_signals:
                y_data = np.real(signal.data)
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
                        signal_path=str(Path(signal_path).relative_to(current_app.config['ROOT_DIR'])),
                        plot_id=plot_id,
                        tab_id=tab_id
                    ),
                    code="""
                    // Send the new slice index to the server
                    fetch(`/signal/update_slice/${signal_path}?tab_id=${tab_id}&plot_id=${plot_id}&slice_index=${cb_obj.value}`)
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


@bp.route('/signal/replace/<path:signal_path>')
def replace(signal_path):
    tab_id = request.args.get('tab_id', '1')  # Default to first tab
    # Get full path
    full_path = current_app.config['ROOT_DIR'] / signal_path
    if not (full_path / '.signal').exists():
        return jsonify({'error': 'Not a signal directory'}), 400

    # Load signal
    signal = Signal.load(full_path)
    tab_state = reset_tab(tab_id)
    plot_state = add_new_plot(tab_state)
    add_signal_to_plot(plot_state, signal)

    return jsonify({'status': 'success'})


@bp.route('/signal/append/<path:signal_path>')
def append(signal_path):
    tab_id = request.args.get('tab_id', '1')  # Default to first tab
    # Get full path
    full_path = current_app.config['ROOT_DIR'] / signal_path
    if not (full_path / '.signal').exists():
        return jsonify({'error': 'Not a signal directory'}), 400

    # Load signal
    signal = Signal.load(full_path)
    tab_state = get_tab(tab_id)
    # Add new plot if none exist, otherwise use last plot
    if tab_state['n_plots'] == 0:
        plot_state = add_new_plot(tab_state)
    else:
        plot_state = get_plot_state(tab_state, tab_state['n_plots'] - 1)
    add_signal_to_plot(plot_state, signal)

    return jsonify({'status': 'success'})

@bp.route('/signal/new_tab/<path:signal_path>')
def new_tab(signal_path):
    # Plot signal in new tab
    tab_id = create_new_tab()
    # Get full path
    full_path = current_app.config['ROOT_DIR'] / signal_path
    if not (full_path / '.signal').exists():
        return jsonify({'error': 'Not a signal directory'}), 400

    # Load signal
    signal = Signal.load(full_path)
    tab_state = get_tab(tab_id)
    # By default, add signal to last plot
    plot_state = add_new_plot(tab_state)
    add_signal_to_plot(plot_state, signal)

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
    # Extract plot_id as int if it starts with "plot_"
    if isinstance(plot_id, str) and plot_id.startswith('plot_'):
        plot_id = int(plot_id[5:])  # Remove "plot_" prefix and convert to int

    slice_index = int(request.args.get('slice_index', 0))

    if tab_id is None or plot_id is None:
        return jsonify({'error': 'Missing tab_id or plot_id'}), 400

    # Get the plot state
    tab_state = get_tab(tab_id)
    plot_state = get_plot_state(tab_state, plot_id)

    # Update slider state
    slider_state = find_slider_state(plot_state, signal_path)
    if slider_state is not None:
        slider_state['current_index'] = slice_index

        # Load the signal
        full_path = current_app.config['ROOT_DIR'] / signal_path
        signal = Signal.load(full_path)

        # Remove old signals for this path
        plot_state['signals'] = [s for s in plot_state['signals']
                               if s['path'] != signal_path]

        # Add new signals with updated slice
        signals_1d, index_tuple_list = signal.to_1d_signals(slice(None), slice_index)
        for sig_1d, index_tuple in zip(signals_1d, index_tuple_list):
            serialized = serialize_signal(sig_1d)
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
    if isinstance(idx, slice):
        return {
            'type': 'slice',
            'start': idx.start,
            'stop': idx.stop,
            'step': idx.step
        }
    return idx

def deserialize_index(idx):
    if isinstance(idx, dict) and idx.get('type') == 'slice':
        return slice(idx['start'], idx['stop'], idx['step'])
    return idx


