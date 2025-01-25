from flask import Blueprint, render_template, current_app, request, jsonify
from pathlib import Path
import json
import numpy as np
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool, CustomJSHover
import re

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Render main page."""
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
        # Get full path
        full_path = current_app.config['ROOT_DIR'] / signal_path
        if not (full_path / '.signal').exists():
            return jsonify({'error': 'Not a signal directory'}), 400

        # Load signal data
        data = np.load(full_path / 'data.npy')
        x_data = np.load(full_path / 'x_data.npy') if (full_path / 'x_data.npy').exists() else np.arange(len(data))

        # Load metadata if exists
        metadata = {}
        if (full_path / 'metadata.json').exists():
            with open(full_path / 'metadata.json') as f:
                metadata = json.load(f)

        # Create figure with tools
        p = figure(
            height=400,
            sizing_mode='stretch_width',
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_drag="pan",
            active_scroll="wheel_zoom",
            title=metadata.get('name', signal_path)
        )

        # Configure axes
        p.xaxis.axis_label = metadata.get('x_name', 'Time')
        p.yaxis.axis_label = metadata.get('unit', '')

        # Add hover tool with generic labels
        x_unit = metadata.get('x_unit', '')
        y_unit = metadata.get('unit', '')

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
                ('y', '@y{custom}' + (f' {y_unit}' if y_unit else ''))
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

        # Create data source
        source = ColumnDataSource({
            'x': x_data,
            'y': data
        })

        # Plot line
        p.line('x', 'y', source=source, line_width=2)

        # Generate plot components
        script, div = components(p)

        # Extract JavaScript
        script_content = re.search(r'<script.*?>(.*?)</script>', script, re.DOTALL)
        if script_content:
            script = script_content.group(1)

        return jsonify({
            'script': script,
            'div': div,
            'metadata': metadata,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error plotting signal: {e}")
        return jsonify({'error': str(e)}), 500
