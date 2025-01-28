# Signal Inspector

A web-based tool for visualizing and analyzing signal data.

## Development Roadmap

### Phase 1: Essential Plot Controls & Real-time Updates
- [ ] Plot Layout Controls
  - [ ] Split/combine traces button in GUI
  - [ ] Independent y-axes with shared x-axis
  - [ ] Combine split traces back to single plot
- [ ] Real-time Updates
  - [ ] Add refresh button
  - [ ] File system watcher for auto-refresh
  - [ ] Visual indicator when data has changed
- [ ] Basic Axis Controls
  - [ ] Log/linear scale toggle buttons
  - [ ] dB scale option
  - [ ] Grid toggle

### Phase 2: Complex Data & Advanced Visualization
- [ ] Complex Signal Support
  - [ ] Detect and indicate complex data in GUI
  - [ ] Plot type selector (real, imag, magnitude, phase)
  - [ ] Phase unwrapping option
- [ ] Heatmap Support
  - [ ] Toggle between trace/heatmap for 2D signals
  - [ ] Colormap selection
  - [ ] Slider controls for 3D+ signals in heatmap mode
  - [ ] Heatmap value tooltip/cursor

### Phase 3: Measurement & Analysis Tools
- [ ] Cursor Tools
  - [ ] Single point measurement
  - [ ] Delta measurements (dx, dy)
  - [ ] Multiple cursors
- [ ] Statistics
  - [ ] Basic stats per trace (min, max, mean, std)
  - [ ] Stats in selected region
  - [ ] Export measurements

### Phase 4: Advanced Interaction & Layout
- [ ] Advanced Plot Layout
  - [ ] Drag & drop traces between plots
  - [ ] Drop zones for split/combine
  - [ ] Plot area resize handles
- [ ] Plot Customization
  - [ ] Trace color/style editor
  - [ ] Legend position/style
  - [ ] Export plot settings

### Phase 5: Additional Features
- [ ] Session Management
  - [ ] Save/load session state
  - [ ] Default configurations
- [ ] Advanced Analysis
  - [ ] FFT/spectrum analysis
  - [ ] Filtering options
  - [ ] Custom math operations
- [ ] Plot Types
  - [ ] Polar plots
  - [ ] Smith charts
  - [ ] Histogram/distribution plots

### Completed Features
- [x] Context Menu for Signal Loading
  - [x] Right-click menu with load options
  - [x] "Add to Current Plot" (overlay)
  - [x] "New Tab"
- [x] Multi-dimensional Signal Support
  - [x] Plot multiple traces with legend
  - [x] Dimension sliders for n-D data
  - [x] Dimension selection UI
- [x] Tab Management
  - [x] Create/close tabs
