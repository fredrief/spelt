# Signal Inspector

A web-based tool for visualizing and analyzing signal data.

## Development Roadmap

### Phase 1: Basic Structure & File Handling
- [x] Set up Flask application structure
- [x] Implement basic file browser
  - [x] Directory navigation
  - [x] Signal file detection
  - [x] Basic UI with Bootstrap
- [x] Add "Open Folder" functionality
- [x] Add navigation controls (up button)
- [x] Filter for signal folders

### Phase 2: Single Plot Functionality
- [x] Basic Bokeh plot integration
  - [x] Single signal display
  - [x] Basic zoom/pan
  - [x] Axes labels
- [x] Signal loading/plotting
  - [x] Units handling
  - [x] SI prefix formatting
- [x] Basic plot interactions
  - [x] Hover tooltips
  - [x] Crosshair

### Phase 3: Multi-Signal & Advanced Plot Layout
- [ ] Context Menu for Signal Loading
  - [ ] Right-click menu with load options
  - [ ] "Add to Current Plot" (overlay)
  - [ ] "New Plot in Tab" (split view)
  - [ ] "New Tab"
- [ ] Multi-dimensional Signal Support
  - [ ] Plot multiple traces with legend
  - [ ] Dimension sliders for n-D data
  - [ ] Dimension selection UI
- [ ] Advanced Plot Layout
  - [ ] Split view with shared x-axis
  - [ ] Drag & drop between plots
  - [ ] Plot area resize handles
- [ ] Tab Management
  - [ ] Create/close tabs
  - [ ] Drag signals between tabs
  - [ ] Tab naming and organization

### Phase 4: Interactive Features
- [ ] Signal Management
  - [ ] Signal visibility toggle
  - [ ] Color/style customization
  - [ ] Signal grouping
- [ ] Drag & Drop Enhancement
  - [ ] Drag to overlay
  - [ ] Drag to split
  - [ ] Drop zone indicators
- [ ] Measurements
  - [ ] Cursor tools
  - [ ] Delta measurements
  - [ ] Statistics per dimension
