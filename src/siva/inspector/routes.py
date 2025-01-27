from flask import Blueprint, render_template, current_app, request, jsonify, session
from pathlib import Path
import json
import numpy as np
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.models import (
    ColumnDataSource, HoverTool, CrosshairTool, CustomJSHover,
    LinearAxis, Range1d, Slider, Column, CustomJS
)
from bokeh.layouts import column
from bokeh.palettes import Category10
import re
from siva.signal import Signal, MetadataKeys, SignalDomain, SamplingType
from enum import Enum

bp = Blueprint('main', __name__)

def serialize_metadata(metadata: dict) -> dict:
    """Convert metadata to JSON serializable format."""
    serialized = {}
    for key, value in metadata.items():
        if isinstance(value, Enum):
            serialized[key] = value.value
        elif isinstance(value, (list, tuple)):
            serialized[key] = [item.value if isinstance(item, Enum) else item for item in value]
        else:
            serialized[key] = value
    return serialized

@bp.route('/')
def index():
    """Render main page."""
    # Initialize tabs state if not exists
    if 'tabs' not in session:
        session['tabs'] = {'1': {  # Default tab
            'signals': [],
            'y_units': [],
            'x_unit': None,
            'x_range': None,
            'y_range': None
        }}
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

@bp.route('/plot/<path:signal_path>')
def plot_signal(signal_path):
    """Plot a signal."""
    try:
        mode = request.args.get('mode', 'replace')
        tab_id = request.args.get('tab_id', '1')  # Default to first tab

        # Get full path
        full_path = current_app.config['ROOT_DIR'] / signal_path
        if not (full_path / '.signal').exists():
            return jsonify({'error': 'Not a signal directory'}), 400

        # Load signal
        signal = Signal.load(full_path)

        # Initialize tabs state if not exists
        if 'tabs' not in session:
            session['tabs'] = {}

        # Initialize plot state for this tab if not exists
        if tab_id not in session['tabs']:
            session['tabs'][tab_id] = {
                'signals': [],
                'y_units': [],
                'x_unit': None,
                'x_range': None,
                'y_range': None,
                'slider_values': {}  # Store slider values per signal
            }
            session.modified = True

        plot_state = session['tabs'][tab_id]

        # If there's no current plot, treat overlay mode as replace
        if mode == 'overlay' and not plot_state['signals']:
            mode = 'replace'
        # If signal is already in plot and mode is overlay, do nothing
        elif mode == 'overlay' and signal_path in plot_state['signals']:
            # Just return success without any changes
            return jsonify({
                'status': 'success',
                'metadata': serialize_metadata(signal.metadata)
            })

        # Check unit compatibility for overlay mode
        if mode == 'overlay' and plot_state['signals']:
            # Check x-axis unit compatibility
            current_x_unit = plot_state['x_unit']
            new_x_unit = signal.metadata.get(MetadataKeys.X_UNIT.value)

            if current_x_unit and new_x_unit and current_x_unit != new_x_unit:
                return jsonify({
                    'error': f'X-axis unit mismatch: {current_x_unit} vs {new_x_unit}'
                }), 400

        # Create new figure
        p = figure(
            height=400,
            sizing_mode='stretch_width',
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_drag="pan",
            active_scroll="wheel_zoom",
            title=signal.metadata.get(MetadataKeys.NAME.value, signal_path)
        )

        # Configure axes
        p.xaxis.axis_label = signal.metadata.get(MetadataKeys.X_NAME.value, 'Time')

        # Get color palette
        colors = Category10[10]

        # Handle y-axis based on mode and units
        y_unit = signal.metadata.get(MetadataKeys.UNIT.value, '')
        y_ranges = {}  # Store y ranges by unit

        # Function to get data slice and calculate ranges
        def get_data_slice(signal, slider_values=None):
            data = signal.data
            if data.ndim > 1:
                # First dimension is always x-axis
                # Second dimension creates multiple traces
                # Higher dimensions use sliders
                if data.ndim > 2:
                    # Create slice tuple for dimensions > 2
                    slice_indices = [slice(None), slice(None)]  # Keep first two dims as is
                    for dim in range(2, data.ndim):
                        idx = slider_values.get(f"{signal_path}_dim_{dim}", 0) if slider_values else 0
                        slice_indices.append(idx)
                    data = data[tuple(slice_indices)]

                # Now data should be 2D (L x M)
                return data
            return data.reshape(-1, 1)  # Make 1D data 2D with shape (L, 1)

        # Calculate ranges for the new signal
        data_slice = get_data_slice(signal, plot_state.get('slider_values', {}))
        x_data = signal.x_data if signal.x_data is not None else np.arange(data_slice.shape[0])
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(np.real(data_slice)), np.max(np.real(data_slice))

        if mode == 'overlay' and plot_state['signals']:
            # First, load all existing signals to calculate ranges
            for existing_path in plot_state['signals']:
                existing_signal = Signal.load(current_app.config['ROOT_DIR'] / existing_path)
                existing_data = get_data_slice(existing_signal, plot_state.get('slider_values', {}))
                existing_x = existing_signal.x_data if existing_signal.x_data is not None else np.arange(existing_data.shape[0])
                x_min = min(x_min, np.min(existing_x))
                x_max = max(x_max, np.max(existing_x))

                existing_y_unit = existing_signal.metadata.get(MetadataKeys.UNIT.value, '')
                if existing_y_unit not in y_ranges:
                    y_ranges[existing_y_unit] = [np.inf, -np.inf]

                y_ranges[existing_y_unit][0] = min(y_ranges[existing_y_unit][0], np.min(existing_data))
                y_ranges[existing_y_unit][1] = max(y_ranges[existing_y_unit][1], np.max(existing_data))

            # Add new signal's range
            if y_unit not in y_ranges:
                y_ranges[y_unit] = [y_min, y_max]
            else:
                y_ranges[y_unit][0] = min(y_ranges[y_unit][0], y_min)
                y_ranges[y_unit][1] = max(y_ranges[y_unit][1], y_max)

            # Set x range with padding
            padding_x = (x_max - x_min) * 0.05
            p.x_range.start = x_min - padding_x
            p.x_range.end = x_max + padding_x

            # Create y ranges for each unit
            for unit, (unit_min, unit_max) in y_ranges.items():
                padding = (unit_max - unit_min) * 0.05  # 5% padding
                if unit == list(y_ranges.keys())[0]:  # First unit uses default range
                    p.y_range.start = unit_min - padding
                    p.y_range.end = unit_max + padding
                    p.yaxis.axis_label = unit
                else:
                    # Create new range for additional units
                    range_name = f'y_{len(p.extra_y_ranges)}'  # Use sequential numbering
                    new_range = Range1d(
                        start=unit_min - padding,
                        end=unit_max + padding,
                        name=range_name
                    )
                    p.extra_y_ranges[range_name] = new_range
                    new_axis = LinearAxis(y_range_name=range_name, axis_label=unit)
                    p.add_layout(new_axis, 'left')

            # Map units to range names for plotting
            unit_to_range = {'default': 'default'}
            for i, unit in enumerate(y_ranges.keys()):
                if i > 0:  # Skip first unit which uses default range
                    unit_to_range[unit] = f'y_{i-1}'
        else:
            # Single signal plot
            padding_x = (x_max - x_min) * 0.05
            padding_y = (y_max - y_min) * 0.05
            p.x_range.start = x_min - padding_x
            p.x_range.end = x_max + padding_x
            p.y_range.start = y_min - padding_y
            p.y_range.end = y_max + padding_y
            p.yaxis.axis_label = y_unit
            unit_to_range = {'default': 'default'}

        # Add hover tool with generic labels
        x_unit = signal.metadata.get(MetadataKeys.X_UNIT.value, '')

        format_function = """
            function format_with_prefix(value, decimals = 3) {
                const prefixes = [
                    {exp: 24, symbol: 'Y'}, {exp: 21, symbol: 'Z'},
                    {exp: 18, symbol: 'E'}, {exp: 15, symbol: 'P'},
                    {exp: 12, symbol: 'T'}, {exp: 9, symbol: 'G'},
                    {exp: 6, symbol: 'M'}, {exp: 3, symbol: 'k'},
                    {exp: 0, symbol: ''},
                    {exp: -3, symbol: 'm'}, {exp: -6, symbol: 'µ'},
                    {exp: -9, symbol: 'n'}, {exp: -12, symbol: 'p'},
                    {exp: -15, symbol: 'f'}, {exp: -18, symbol: 'a'},
                    {exp: -21, symbol: 'z'}, {exp: -24, symbol: 'y'}
                ];

                if (value === 0) {
                    return `0${decimals === 0 ? '' : '.' + '0'.repeat(decimals)}`;
                }

                const exp = Math.floor(Math.log10(Math.abs(value)) / 3) * 3;
                const clampedExp = Math.min(Math.max(exp, -24), 24);

                const prefix = prefixes.find(p => p.exp === clampedExp) || {symbol: ''};
                const scaledValue = value * Math.pow(10, -clampedExp);

                return `${scaledValue.toFixed(decimals)}${prefix.symbol}`;
            }
            return format_with_prefix(value, 3);
        """

        hover = HoverTool(
            tooltips=[
                ('x', '@x{custom}' + (f' {x_unit}' if x_unit else '')),
                ('y', '@y{custom}' + (f' {y_unit}' if y_unit else '')),
                ('trace', '@legend')
            ],
            formatters={
                '@x': CustomJSHover(code=format_function),
                '@y': CustomJSHover(code=format_function)
            },
            mode='vline'
        )
        p.add_tools(hover)

        # Add crosshair
        crosshair = CrosshairTool(
            line_color='gray',
            line_alpha=0.5,
            line_width=1
        )
        p.add_tools(crosshair)

        # Add grid
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_alpha = 0.3
        p.grid.minor_grid_line_color = 'gray'
        p.grid.minor_grid_line_alpha = 0.1

        # Plot all signals in overlay mode
        if mode == 'overlay':
            # Plot existing signals first
            for i, existing_path in enumerate(plot_state['signals']):
                existing_signal = Signal.load(current_app.config['ROOT_DIR'] / existing_path)
                existing_y_unit = existing_signal.metadata.get(MetadataKeys.UNIT.value, '')
                existing_data = get_data_slice(existing_signal, plot_state.get('slider_values', {}))

                # Use the first unit's range as default, otherwise use the mapped range
                if existing_y_unit == list(y_ranges.keys())[0]:
                    range_name = 'default'
                else:
                    range_name = unit_to_range[existing_y_unit]

                # Plot each trace from the second dimension
                for j in range(existing_data.shape[1]):
                    source = ColumnDataSource({
                        'x': existing_signal.x_data if existing_signal.x_data is not None else np.arange(existing_data.shape[0]),
                        'y': np.real(existing_data[:, j]),
                        'legend': [f'Trace {j+1}' for _ in range(existing_data.shape[0])]
                    })
                    legend_label = f"{existing_signal.metadata.get(MetadataKeys.NAME.value, 'Signal')} - Trace {j+1}"
                    p.line('x', 'y', source=source, line_width=2, y_range_name=range_name,
                          color=colors[(i * existing_data.shape[1] + j) % len(colors)],
                          legend_label=legend_label)

            # Plot the new signal with the next color
            color_index = len(plot_state['signals']) * data_slice.shape[1] % len(colors)
        else:
            color_index = 0

        # Create sliders for dimensions > 2
        sliders = []
        sources = []  # Store all data sources for updating
        if signal.data.ndim > 2:
            # Create separate data sources for each trace
            sources = []
            for j in range(data_slice.shape[1]):
                source = ColumnDataSource({
                    'x': signal.x_data if signal.x_data is not None else np.arange(data_slice.shape[0]),
                    'y': np.real(data_slice[:, j]),
                    'legend': [f'Trace {j+1}'] * data_slice.shape[0]  # One label per trace
                })
                sources.append(source)

            # Store the full data for JavaScript
            full_data = np.real(signal.data)  # Store only real part
            js_data = full_data.tolist()  # Convert to list for JSON serialization

            for dim in range(2, signal.data.ndim):
                dim_name = signal.metadata.get(MetadataKeys.DIMENSIONS.value, [])[dim] if dim < len(signal.metadata.get(MetadataKeys.DIMENSIONS.value, [])) else f'Dimension {dim+1}'
                slider = Slider(
                    start=0,
                    end=signal.data.shape[dim] - 1,
                    value=plot_state.get('slider_values', {}).get(f"{signal_path}_dim_{dim}", 0),
                    step=1,
                    title=dim_name,
                    width=p.width  # Match plot width
                )
                sliders.append(slider)

            # Create JavaScript callback for slider updates
            callback_code = """
                // These variables are passed in through args
                const data = full_data;
                const data_sources = sources;
                const all_sliders = sliders;

                // Update each source
                for (let i = 0; i < data_sources.length; i++) {
                    let trace_data = data.map(row => row[i]);  // Get data for this trace

                    // Apply higher dimension indices
                    for (let dim = 0; dim < all_sliders.length; dim++) {
                        const idx = all_sliders[dim].value;
                        trace_data = trace_data.map(val => val[idx]);
                    }

                    data_sources[i].data.y = trace_data;
                    data_sources[i].change.emit();
                }
            """

            callback = CustomJS(
                args=dict(
                    sources=sources,
                    full_data=js_data,
                    sliders=sliders
                ),
                code=callback_code
            )

            # Connect callback to all sliders
            for slider in sliders:
                slider.js_on_change('value', callback)

            # Plot using the data sources
            for j, source in enumerate(sources):
                p.line(
                    'x', 'y',
                    source=source,
                    line_width=2,
                    color=colors[j % len(colors)],
                    legend_label=f"{signal.metadata.get(MetadataKeys.NAME.value, 'Signal')} - Trace {j+1}"
                )
        else:
            # Original plotting code for 1D/2D signals
            for j in range(data_slice.shape[1]):
                source = ColumnDataSource({
                    'x': signal.x_data if signal.x_data is not None else np.arange(data_slice.shape[0]),
                    'y': np.real(data_slice[:, j]),
                    'legend': [f'Trace {j+1}'] * data_slice.shape[0]  # One label per trace
                })
                sources.append(source)

                legend_label = f"{signal.metadata.get(MetadataKeys.NAME.value, 'Signal')} - Trace {j+1}"
                p.line('x', 'y', source=source, line_width=2,
                      color=colors[j % len(colors)],
                      legend_label=legend_label)

        # Configure legend
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        p.legend.background_fill_alpha = 0.7

        # Create layout with proper sizing
        if sliders:
            # Create a column layout with proper sizing
            layout = column(
                p,
                *sliders,
                sizing_mode='stretch_width',  # Make everything stretch to container width
                styles={'width': '100%'}  # Ensure full width
            )
            script, div = components(layout)
        else:
            script, div = components(p)

        # Extract JavaScript
        script_content = re.search(r'<script.*?>(.*?)</script>', script, re.DOTALL)
        if script_content:
            script = script_content.group(1)

        # Return plot data
        return jsonify({
            'script': script,
            'div': div,
            'metadata': serialize_metadata(signal.metadata),
            'status': 'success'
        })

    except Exception as e:
        print(f"Error plotting signal: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/clear-plot-state')
def clear_plot_state():
    """Clear the plot state."""
    tab_id = request.args.get('tab_id')
    if tab_id:
        # Clear specific tab
        if 'tabs' in session and tab_id in session['tabs']:
            session['tabs'].pop(tab_id)
            session.modified = True
    else:
        # Clear all tabs
        if 'tabs' in session:
            session.pop('tabs')
    return jsonify({'status': 'success'})
