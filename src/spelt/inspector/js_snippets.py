"""JavaScript code snippets used in routes.py"""

# Format function for hover tool tooltips
FORMAT_WITH_PREFIX = '''
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
'''

# Keyboard shortcut handler for zoom tools
ZOOM_KEY_HANDLER = '''
// Add global key event listeners
if (!window._keyListenersAdded) {
    window.addEventListener('keydown', function(e) {
        // Handle zoom shortcuts
        if (e.shiftKey) {
            wx.active = true;
            wy.active = false;
        } else if (e.ctrlKey || e.metaKey) {
            wx.active = false;
            wy.active = true;
        }
    }, true);  // Use capture phase to intercept before browser

    window.addEventListener('keyup', function(e) {
        if (!e.shiftKey && !e.ctrlKey && !e.metaKey) {
            wx.active = false;
            wy.active = false;
        }
    });

    window._keyListenersAdded = true;
}
'''

# Slider update callback
SLIDER_UPDATE_CALLBACK = '''
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
'''
