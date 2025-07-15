"""
FTS Toolkit Pressure Gradient Analysis Module

This module provides comprehensive pressure gradient analysis capabilities for formation
testing operations. It enables advanced pressure-depth analysis, fluid type determination,
gradient calculations, and integrated log data visualization.

CORE FUNCTIONALITY:
- Pressure vs. depth analysis with linear regression
- Multi-zone fluid gradient calculations
- Formation pressure extrapolation and interpretation
- LAS log data integration and visualization
- Interactive point selection and data manipulation
- Comprehensive unit conversion and depth handling

ANALYSIS CAPABILITIES:
- Linear regression for pressure gradient determination
- Statistical analysis with correlation coefficients
- Multi-point pressure extrapolation
- Formation fluid type identification (oil, gas, water)
- Pressure communication analysis between zones
- Contamination assessment through gradient analysis

VISUALIZATION FEATURES:
- Interactive Plotly charts with zoom and pan capabilities
- Dual-axis plotting for pressure and log curves
- Customizable depth and pressure ranges
- Professional styling optimized for technical presentations
- Real-time updates with user interactions
- Export capabilities for reports and presentations

DATA MANAGEMENT:
- Persistent state management with JSON serialization
- LAS file integration for comprehensive analysis
- Flexible unit handling (feet/meters, psi/bar)
- Data validation and error handling
- Session persistence across application restarts

INDUSTRY CONTEXT:
Pressure gradient analysis is fundamental to formation evaluation and completion design.
This application handles critical workflows including:
- Formation pressure prediction for drilling safety
- Fluid contact identification and mapping
- Reservoir connectivity assessment
- Completion interval optimization
- Pressure depletion monitoring
- Formation testing data interpretation

TECHNICAL ARCHITECTURE:
- Class-based design with state encapsulation
- Dataclass structures for type safety and serialization
- Event-driven UI updates with Flet framework
- Plotly integration for professional visualization
- Modular functions for mathematical computations
- Comprehensive error handling and user feedback

DEPENDENCIES:
- flet: Modern cross-platform UI framework
- plotly: Interactive scientific plotting
- pandas/numpy: Data manipulation and numerical analysis
- lasio: LAS file parsing and integration
- dataclasses: Type-safe data structures
- pathlib: Modern file system operations
 
AUTHOR: FTS Engineering Team
VERSION: 2.0
LAST_UPDATED: 2025
"""

# ====================================================================
# CORE IMPORTS AND DEPENDENCIES
# ====================================================================

# Standard library imports for data handling and system operations
import base64
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from io import StringIO
from typing import List, Optional
import logging
import pathlib



# Third-party imports for UI and numerical analysis
import flet as ft
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

# ====================================================================
# CONFIGURATION AND SETUP
# ====================================================================


# Configure application paths for logging and data storage
LOG_PATH = pathlib.Path(__file__).parent.parent / "logs" / "pressure_gradient_app.log"
DATA_PATH = pathlib.Path(__file__).parent.parent / "storage" / "data"

# Ensure logs directory exists
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configure module-specific logging (per-page log file)
logger = logging.getLogger("pressure_gradient_app")
if not logger.hasHandlers():
    file_handler = logging.FileHandler(LOG_PATH)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# ====================================================================
# BRAND IDENTITY AND STYLING
# ====================================================================

# Professional color palette optimized for technical presentations
# Colors selected specifically for formation testing industry standards
BRAND_DARK_GREEN = "#1B4332"  # Deep green for headers and emphasis
BRAND_GREEN = "#2D6A4F"  # Primary brand green for key elements
BRAND_LIGHT_GREEN = "#95D5B2"  # Light green for backgrounds and highlights
BRAND_YELLOW = "#FFD166"  # Attention color for warnings and alerts
BRAND_GRAY = "#6C757D"  # Neutral gray for secondary information
BRAND_LIGHT_GRAY = "#F8F9FA"  # Background color for panels and containers
BRAND_WHITE = "#FFFFFF"  # Pure white for contrast and readability
BRAND_ACCENT_BLUE = "#118AB2"  # Professional blue for primary actions
BRAND_ACCENT_ORANGE = "#EF476F"  # Alert orange for critical information
BRAND_ACCENT_PURPLE = "#7209B7"  # Purple for special categories and highlights

# Derived colors for specific UI applications
BRAND_BG_COLOR = BRAND_LIGHT_GRAY  # Main application background
BRAND_SECONDARY_BG_COLOR = "#E9F5F0"  # Secondary panel backgrounds
BRAND_BORDER_COLOR = BRAND_LIGHT_GREEN  # Border elements and dividers
BRAND_CHART_BORDER_COLOR = "#D8EAE2"  # Chart container borders
BRAND_DARK_GRAY = "#495057"  # Dark text and emphasis elements


# ====================================================================
# DATA STRUCTURES AND MODELS
# ====================================================================


@dataclass
class PressurePoint:
    """
    Represents a single pressure measurement point in formation testing analysis.

    This dataclass encapsulates the essential information for each pressure measurement,
    including spatial location, measured value, and metadata required for gradient
    analysis and visualization.

    Attributes:
        zone (str): Formation zone or interval name for organizational purposes
        tvd (float): True Vertical Depth of the measurement point
        pressure (float): Measured pressure value at the specified depth
        selected (bool): Selection state for analysis inclusion/exclusion

    Industry Context:
        - TVD critical for accurate gradient calculations
        - Zone names help organize measurements by formation intervals
        - Selection state allows excluding outliers or suspect data points
        - Pressure values typically from formation testing tools (MDT, RCI, etc.)

    Usage Examples:
        - PressurePoint("Sand A", 8450.5, 3876.2, True)
        - PressurePoint("Shale B", 8521.0, 3901.8, False)  # Excluded from analysis
    """

    zone: str
    tvd: float
    pressure: float
    selected: bool = True


@dataclass
class AppState:
    """
    Complete application state container for persistence and session management.

    This dataclass captures all user inputs, analysis parameters, and application
    state to enable session persistence across application restarts and provide
    comprehensive state management.

    State Categories:
        - Pressure Data: Measurement points and selections
        - Log Integration: LAS file data and curve selections
        - Display Settings: Units, zoom ranges, and visualization parameters
        - Metadata: Well information and session tracking

    Persistence Features:
        - JSON serializable for file-based storage
        - Complete state restoration capability
        - Incremental state updates during user interactions
        - Backward compatibility with previous versions

    Analysis Parameters:
        - Unit selections affect all calculations and displays
        - Zoom ranges control visualization focus areas
        - Curve selections determine log data overlay
        - Auto-zoom settings provide optimal data viewing
    """

    pressure_points: List[dict]  # Serialized pressure measurement data
    log_data: Optional[str] = None  # Base64 encoded LAS file data
    tvd_column: Optional[str] = None  # Selected depth column from LAS file
    wellname: Optional[str] = None  # Well identifier for reporting
    curves: Optional[List[str]] = None  # Available curves from LAS file
    selected_curve: Optional[str] = None  # Currently selected curve for overlay
    depth_unit: str = "ft"  # Depth measurement unit (ft/m)
    pressure_unit: str = "psi"  # Pressure measurement unit (psi/bar)
    zoom_min: str = ""  # Minimum depth for zoom range
    zoom_max: str = ""  # Maximum depth for zoom range
    zoom_pressure_min: str = ""  # Minimum pressure for zoom range
    zoom_pressure_max: str = ""  # Maximum pressure for zoom range
    auto_zoom: bool = True  # Automatic zoom to data extent
    last_saved: str = ""  # Timestamp of last state save


# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================


def convert_depth(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert depth values between different measurement units.

    This function handles the common conversion between feet and meters used
    in formation testing and well logging operations. Essential for international
    projects and data integration from different measurement systems.

    Args:
        value (float): Numeric depth value to convert
        from_unit (str): Source unit ("ft" or "m")
        to_unit (str): Target unit ("ft" or "m")

    Returns:
        float: Converted depth value in target units

    Conversion Factors:
        - Feet to meters: divide by 3.28084
        - Meters to feet: multiply by 3.28084
        - Same unit: return original value unchanged

    Industry Standards:
        - US operations typically use feet
        - International operations often use meters
        - LAS files may contain data in either unit system
        - Consistent unit handling critical for accurate analysis

    Error Handling:
        - Returns original value for unrecognized units
        - Handles same-unit conversions efficiently
        - Maintains precision for engineering calculations
    """
    if from_unit == to_unit:
        return value
    if from_unit == "m" and to_unit == "ft":
        return value * 3.28084
    if from_unit == "ft" and to_unit == "m":
        return value / 3.28084
    return value


def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """Convert pressure values between different units (psi and bar)."""
    if from_unit == to_unit:
        return value
    if from_unit == "bar" and to_unit == "psi":
        return value * 14.5038
    if from_unit == "psi" and to_unit == "bar":
        return value / 14.5038
    return value


def save_points_to_file(
    points: List[PressurePoint], filename: str = "pressure_points.json"
):
    """Save pressure points to a JSON file"""
    try:
        data = []
        for point in points:
            data.append(
                {
                    "zone": getattr(point, "zone", ""),
                    "tvd": point.tvd,
                    "pressure": point.pressure,
                    "selected": point.selected,
                }
            )

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except (IOError, json.JSONDecodeError) as e:
        logger.error("Error saving points: %s", str(e))
        return False


def load_points_from_file(
    filename: str = "pressure_points.json",
) -> List[PressurePoint]:
    """Load pressure points from a JSON file"""
    try:
        if not os.path.exists(filename):
            return []

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = []
        for item in data:
            # Support legacy files without zone
            zone_val = item.get("zone", "")
            if all(key in item for key in ["tvd", "pressure", "selected"]):
                points.append(
                    PressurePoint(
                        zone=zone_val,
                        tvd=item["tvd"],
                        pressure=item["pressure"],
                        selected=item.get("selected", True),
                    )
                )
            else:
                logger.warning("Skipping invalid item in JSON: %s", item)
        return points
    except (IOError, json.JSONDecodeError, KeyError) as e:
        logger.error("Error loading points: %s", str(e))
        return []




def save_app_state(
    points: List[PressurePoint],
    page: ft.Page,
    depth_unit_val: str,
    pressure_unit_val: str,
    zoom_controls: dict,
    curve_dropdown_val: str,
    filename: str = "app_state.json",
):
    """Save complete application state"""
    try:
        # Convert points to dictionaries
        points_data = [asdict(point) for point in points]

        # Get stored LAS data
        log_data = page.client_storage.get("log_data")
        tvd_column = page.client_storage.get("tvd_column")
        wellname = page.client_storage.get("wellname")
        curves = page.client_storage.get("curves_list")

        state = AppState(
            pressure_points=points_data,
            log_data=log_data,
            tvd_column=tvd_column,
            wellname=wellname,
            curves=curves or [],
            selected_curve=curve_dropdown_val,
            depth_unit=depth_unit_val,
            pressure_unit=pressure_unit_val,
            zoom_min=zoom_controls.get("min", ""),
            zoom_max=zoom_controls.get("max", ""),
            zoom_pressure_min=zoom_controls.get("pressure_min", ""),
            zoom_pressure_max=zoom_controls.get("pressure_max", ""),
            auto_zoom=zoom_controls.get("auto", True),
            last_saved=datetime.now().isoformat(),
        )

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)

        logger.info("App state saved at %s", datetime.now())
        return True
    except (IOError, json.JSONDecodeError) as e:
        logger.error("Error saving app state: %s", str(e))
        return False


def load_app_state(filename: str = "app_state.json") -> Optional[AppState]:
    """Load complete application state"""
    try:
        if not os.path.exists(filename):
            return None

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = AppState(**data)
        logger.info("App state loaded from %s", state.last_saved)
        return state
    except (IOError, json.JSONDecodeError, KeyError) as e:
        logger.error("Error loading app state: %s", str(e))
        return None



def create_empty_plot():
    # Returns a placeholder for the Flet chart area
    return ft.Container(ft.Text("Add pressure points to view plot", color=BRAND_GRAY, size=18), alignment=ft.alignment.center, height=400)

def render_matplotlib_plot(points, stats):
    fig, ax = plt.subplots(figsize=(8, 6))
    if not points:
        ax.text(0.5, 0.5, "Add pressure points to view plot", ha="center", va="center", fontsize=14, color="gray")
        ax.axis('off')
    else:
        all_tvd = [p.tvd for p in points]
        all_pressure = [p.pressure for p in points]
        selected = [p.selected for p in points]
        # Plot all points
        ax.scatter(all_pressure, all_tvd, c=["green" if s else "gray" for s in selected], s=60, label="Points")
        # Plot regression line if available
        if stats:
            (slope, intercept), *_ = stats
            tvd_range = np.array([min(all_tvd), max(all_tvd)])
            pressure_fit = slope * tvd_range + intercept
            ax.plot(pressure_fit, tvd_range, color="#118AB2", linewidth=2, label="Gradient Fit")
        ax.set_xlabel("Pressure (psi)")
        ax.set_ylabel("TVD (ft)")
        ax.set_title("Pressure vs TVD")
        ax.invert_yaxis()
        ax.grid(True)
        ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64



# Removed: figure_to_base64 (no longer needed)


def show_banner(page: ft.Page, message: str, is_error: bool = False):
    """Show a banner message to the user"""
    snack_bar = ft.SnackBar(
        content=ft.Text(message, color=BRAND_WHITE),
        bgcolor=ft.Colors.RED_400 if is_error else BRAND_GREEN,
        action="OK",
        action_color=BRAND_WHITE,
    )
    page.overlay.append(snack_bar)
    snack_bar.open = True
    page.update()


def clear_all_state_files():
    """Clear all application state files"""
    state_files = [
        "app_state.json",
        "pressure_points.json",
        "time_app_state.json",
        "fluid_sampling_state.json",
    ]

    cleared_count = 0
    for filename in state_files:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                cleared_count += 1
                logger.info("Removed state file: %s", filename)
        except (OSError, IOError) as e:
            logger.error("Error removing %s: %s", filename, str(e))

    logger.info("Cleared %d state files", cleared_count)
    return cleared_count > 0


def clear_client_storage(page: ft.Page):
    """Clear all client storage data"""
    try:
        # Clear log data storage
        page.client_storage.clear()
        logger.info("Client storage cleared")
        return True
    except (AttributeError, KeyError) as e:
        logger.error("Error clearing client storage: %s", str(e))
        return False


def calculate_gradient_statistics(selected_points):
    """Calculate gradient and additional statistics"""
    if len(selected_points) < 2:
        logger.debug("Not enough points for gradient calculation")
        return None

    tvd = np.array([p.tvd for p in selected_points])
    pressure = np.array([p.pressure for p in selected_points])

    # Use scipy.stats for more detailed regression statistics
    from scipy.stats import (
        linregress,
    )  # Calculate linear regression with confidence intervals

    gradient_result = linregress(tvd, pressure)
    slope, intercept, r_value, _, std_err = gradient_result  # type: ignore

    logger.debug("Calculated gradient slope: %.4f psi/ft", slope)

    # Calculate R-squared (goodness of fit)
    r_squared = r_value**2  # type: ignore

    # Calculate 95% confidence intervals for the slope
    n = len(tvd)

    # Calculate confidence intervals (t-distribution with n-2 degrees of freedom)
    from scipy.stats import t

    t_critical = t.ppf(0.975, n - 2)  # 95% CI requires 0.975 (two-tailed)
    ci_range = t_critical * std_err  # type: ignore

    lower_ci = slope - ci_range  # type: ignore
    upper_ci = slope + ci_range  # type: ignore

    # Convert slope to density in ppg
    # Correct conversion: 1 psi/ft = 2.31 ppg (pounds per gallon)
    density_ppg = slope * 2.31  # type: ignore
    logger.debug("Density in ppg: %.3f", density_ppg)

    # Convert ppg to g/cc: 1 ppg = 0.1198 g/cc
    density_gcc = density_ppg * 0.1198
    logger.debug("Density in g/cc: %.3f", density_gcc)

    # Determine expected fluid type based on gradient
    fluid_type = determine_fluid_type(slope)
    logger.debug("Final fluid_type returned: %s", fluid_type)

    return (
        (slope, intercept),
        density_ppg,
        density_gcc,
        r_squared,
        std_err,
        lower_ci,
        upper_ci,
        fluid_type,
    )


def determine_fluid_type(gradient_psi_ft):
    """Determine likely fluid type based on gradient value - using exact boundaries"""
    logger.debug(
        "determine_fluid_type called with gradient: %.4f psi/ft", gradient_psi_ft
    )

    if 0.040 <= gradient_psi_ft <= 0.210:
        fluid_type = "Gas"
    elif 0.210 < gradient_psi_ft <= 0.430:
        fluid_type = "Oil"
    elif 0.430 < gradient_psi_ft <= 0.563:
        fluid_type = "Water"
    else:
        # Handle values outside the defined ranges
        if gradient_psi_ft < 0.040:
            fluid_type = "Gas (very light)"
        elif gradient_psi_ft > 0.563:
            fluid_type = "Water (heavy/overpressured)"
        else:
            fluid_type = "Unknown"

    logger.debug("Determined fluid type: %s", fluid_type)
    return fluid_type


class PressureGradientApp:
    """Pressure Gradient Analysis Application - class-based implementation"""

    def __init__(self, page: ft.Page, shared_data: dict):
        self.page = page
        self.shared_data = shared_data

        # App State
        self.points: List[PressurePoint] = []

        # UI Controls (will be initialized in create_ui_controls)
        self.tvd_input: ft.TextField
        self.pressure_input: ft.TextField
        self.zone_input: ft.TextField
        self.depth_unit: ft.Dropdown
        self.pressure_unit: ft.Dropdown
        self.table: ft.DataTable
        # Removed: self.plotly_chart (no longer needed)
        # Removed: self.curve_dropdown, self.file_picker
        self.bulk_dialog: ft.AlertDialog
        self.tvd_area: ft.TextField
        self.pressure_area: ft.TextField
        self.zone_area: ft.TextField
        self.zoom_min_field: ft.TextField
        self.zoom_max_field: ft.TextField
        self.zoom_pressure_min_field: ft.TextField
        self.zoom_pressure_max_field: ft.TextField
        self.auto_zoom_checkbox: ft.Checkbox

        # Load initial state
        self.load_initial_state()

        # Create UI controls
        self.create_ui_controls()

        # Initialize table and plot
        self.update_table()
        self.update_plot()

    def load_initial_state(self):
        """Load initial state from saved data"""
        saved_state = load_app_state()
        if saved_state and saved_state.pressure_points:
            self.points.extend(
                PressurePoint(
                    zone=p.get("zone", ""),
                    tvd=p["tvd"],
                    pressure=p["pressure"],
                    selected=p.get("selected", True),
                )
                for p in saved_state.pressure_points
            )
            self.shared_data["pressure_points"] = self.points

        self.saved_state = saved_state

    def create_ui_controls(self):
        """Create and initialize all UI controls"""
        initial_depth_unit = self.saved_state.depth_unit if self.saved_state else "ft"
        initial_pressure_unit = (
            self.saved_state.pressure_unit if self.saved_state else "psi"
        )
        initial_auto_zoom = self.saved_state.auto_zoom if self.saved_state else True

        # Larger touch-friendly fields for mobile
        self.tvd_input = ft.TextField(
            label="TVD",
            width=160,
            height=48,
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            dense=False,
            text_size=13,
            label_style=ft.TextStyle(size=11),
        )
        self.pressure_input = ft.TextField(
            label="Pressure",
            width=160,
            height=48,
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            dense=False,
            text_size=13,
            label_style=ft.TextStyle(size=11),
        )
        self.zone_input = ft.TextField(
            label="Zone (Optional)",
            width=180,
            height=48,
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            dense=False,
            text_size=13,
            label_style=ft.TextStyle(size=11),
        )
        self.depth_unit = ft.Dropdown(
            width=120,
            options=[
                ft.dropdown.Option("ft", text="FT"),
                ft.dropdown.Option("m", text="M"),
            ],
            value=initial_depth_unit,
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            on_change=self.update_table_and_page,
            dense=False,
            text_size=13,
        )
        self.pressure_unit = ft.Dropdown(
            width=120,
            options=[
                ft.dropdown.Option("psi", text="Psi"),
                ft.dropdown.Option("bar", text="Bar"),
            ],
            value=initial_pressure_unit,
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            on_change=self.update_table_and_page,
            dense=False,
            text_size=13,
        )
        self.zoom_min_field = ft.TextField(
            width=120,
            height=44,
            value=self.saved_state.zoom_min if self.saved_state else "",
            label="Min TVD",
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            dense=False,
            text_size=13,
            label_style=ft.TextStyle(size=11),
        )
        self.zoom_max_field = ft.TextField(
            width=120,
            height=44,
            value=self.saved_state.zoom_max if self.saved_state else "",
            label="Max TVD",
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            dense=False,
            text_size=13,
            label_style=ft.TextStyle(size=11),
        )
        self.zoom_pressure_min_field = ft.TextField(
            width=120,
            height=44,
            value=self.saved_state.zoom_pressure_min if self.saved_state else "",
            label="Min Pressure",
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            dense=False,
            text_size=13,
            label_style=ft.TextStyle(size=11),
        )
        self.zoom_pressure_max_field = ft.TextField(
            width=120,
            height=44,
            value=self.saved_state.zoom_pressure_max if self.saved_state else "",
            label="Max Pressure",
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            dense=False,
            text_size=13,
            label_style=ft.TextStyle(size=11),
        )
        self.auto_zoom_checkbox = ft.Checkbox(
            label="Auto Zoom", value=initial_auto_zoom, check_color=BRAND_GREEN
        )
        self.table = ft.DataTable(
            columns=[
                ft.DataColumn(
                    ft.Text(
                        "Zone",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=BRAND_WHITE,
                    )
                ),
                ft.DataColumn(
                    ft.Text(
                        f"TVD ({initial_depth_unit})",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=BRAND_WHITE,
                    )
                ),
                ft.DataColumn(
                    ft.Text(
                        f"Pressure ({initial_pressure_unit})",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=BRAND_WHITE,
                    )
                ),
                ft.DataColumn(
                    ft.Text(
                        "Selected",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=BRAND_WHITE,
                    )
                ),
                ft.DataColumn(
                    ft.Text(
                        "Delete", size=12, weight=ft.FontWeight.BOLD, color=BRAND_WHITE
                    )
                ),
            ],
            rows=[],
            border=ft.border.all(1, BRAND_GREEN),
            border_radius=8,
            vertical_lines=ft.border.BorderSide(1, BRAND_GREEN),
            horizontal_lines=ft.border.BorderSide(1, BRAND_GREEN),
            heading_row_color=BRAND_GREEN,
            heading_row_height=40,
            data_row_color={ft.ControlState.HOVERED: BRAND_SECONDARY_BG_COLOR},
            data_row_min_height=35,
            data_row_max_height=35,
            column_spacing=20,
            divider_thickness=1,
        )
        self.curve_dropdown = ft.Dropdown(
            width=200,
            label="Select Log Curve",
            disabled=True,
            bgcolor=BRAND_BG_COLOR,
            border_color=BRAND_BORDER_COLOR,
            on_change=self.update_plot,
        )
        # Removed: file_picker and related overlay

        self.tvd_area = ft.TextField(
            label="Paste TVD values (one per line)",
            multiline=True,
            min_lines=4,
            max_lines=6,
            width=220,
            expand=True,
            dense=False,
        )
        self.pressure_area = ft.TextField(
            label="Paste pressure values (one per line)",
            multiline=True,
            min_lines=4,
            max_lines=6,
            width=220,
            expand=True,
            dense=False,
        )
        self.zone_area = ft.TextField(
            label="Paste Zone Names (one per line, optional)",
            multiline=True,
            min_lines=4,
            max_lines=6,
            width=220,
            expand=True,
            dense=False,
        )
        self.plot_container = ft.Container(content=create_empty_plot(), expand=True, padding=8)
        # Statistics card (will be updated in update_statistics_card)
        self.statistics_card = ft.Card(
            content=ft.Container(
                content=ft.Text("Statistics will appear here.", size=16, color=BRAND_DARK_GRAY),
                bgcolor=BRAND_SECONDARY_BG_COLOR,
                padding=ft.padding.all(16),
            ),
            elevation=2,
            margin=ft.margin.only(top=10, bottom=10),
        )
        self.bulk_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Bulk Data Input"),
            content=ft.Column(
                [
                    ft.Text(
                        "Paste data below. Ensure TVD and Pressure have the same number of lines.", size=14
                    ),
                    ft.Row([self.tvd_area, self.pressure_area, self.zone_area], spacing=8),
                ],
                tight=True,
            ),
            actions=[
                ft.TextButton("Add Data", on_click=self.process_bulk_data, style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=16, vertical=10))),
                ft.TextButton("Cancel", on_click=self.close_bulk_dialog, style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=16, vertical=10))),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

    def update_table_and_page(self, _=None):
        """Update table and refresh the page"""
        self.update_table()
        self.page.update()

    def update_plot_and_page(self, _=None):
        """Update plot and refresh the page"""
        self.update_plot()
        self.page.update()

    def update_table(self):
        """Update the data table with the current points"""
        current_depth_unit = self.depth_unit.value or "ft"
        current_pressure_unit = self.pressure_unit.value or "psi"

        # Update table headers
        self.table.columns[1].label = ft.Text(
            f"TVD ({current_depth_unit})",
            size=12,
            weight=ft.FontWeight.BOLD,
            color=BRAND_WHITE,
        )
        self.table.columns[2].label = ft.Text(
            f"Pressure ({current_pressure_unit})",
            size=12,
            weight=ft.FontWeight.BOLD,
            color=BRAND_WHITE,
        )

        if hasattr(self.table, "rows") and self.table.rows is not None:
            self.table.rows.clear()
            for i, point in enumerate(self.points):
                display_tvd = convert_depth(point.tvd, "ft", current_depth_unit)
                display_pressure = convert_pressure(
                    point.pressure, "psi", current_pressure_unit
                )

                self.table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(point.zone)),
                            ft.DataCell(ft.Text(f"{display_tvd:.2f}")),
                            ft.DataCell(ft.Text(f"{display_pressure:.2f}")),
                            ft.DataCell(
                                ft.Checkbox(
                                    value=point.selected,
                                    data=i,
                                    on_change=self.toggle_point_selection,
                                )
                            ),
                            ft.DataCell(
                                ft.IconButton(
                                    icon=ft.Icons.DELETE,
                                    icon_color=BRAND_ACCENT_ORANGE,
                                    data=i,
                                    on_click=self.delete_point,
                                    tooltip="Delete Point",
                                )
                            ),
                        ]
                    )
                )

    def update_plot(self, _=None):
        """Update the plot with the current data using matplotlib and display as image in Flet"""
        selected_points = [p for p in self.points if p.selected]
        stats = calculate_gradient_statistics(selected_points)

        if not self.points:
            self.plot_container.content = create_empty_plot()
        else:
            img_base64 = render_matplotlib_plot(self.points, stats)
            self.plot_container.content = ft.Image(src_base64=img_base64, width=800, height=500, fit=ft.ImageFit.CONTAIN)

        self.update_statistics_card(stats, selected_points)
        self.page.update()

    def update_statistics_card(self, stats, selected_points):
        if not self.points:
            self.statistics_card.content = ft.Container(
                content=ft.Text("Statistics will appear here.", size=14, color=BRAND_DARK_GRAY),
                bgcolor=BRAND_SECONDARY_BG_COLOR,
                padding=ft.padding.all(12),
            )
        elif stats:
            (slope, intercept), density_ppg, density_gcc, r_squared, _, lower_ci, upper_ci, fluid_type = stats
            stats_text = (
                f"Fluid Gradient: {slope:.4f} psi/ft\n"
                f"95% CI: [{lower_ci:.4f}, {upper_ci:.4f}]\n"
                f"Fluid Density: {density_ppg:.2f} ppg | {density_gcc:.2f} g/cc\n"
                f"R-squared: {r_squared:.4f}\n"
                f"Fluid Type: {fluid_type}"
            )
            self.statistics_card.content = ft.Container(
                content=ft.Text(stats_text, size=15, color=BRAND_DARK_GRAY, selectable=True),
                bgcolor=BRAND_SECONDARY_BG_COLOR,
                padding=ft.padding.all(12),
            )
        elif len(selected_points) < 2:
            self.statistics_card.content = ft.Container(
                content=ft.Text("Select at least two points to calculate gradient statistics.", size=14, color=BRAND_ACCENT_ORANGE),
                bgcolor=BRAND_SECONDARY_BG_COLOR,
                padding=ft.padding.all(12),
            )
        else:
            self.statistics_card.content = ft.Container(
                content=ft.Text("Unable to calculate statistics. Check your data.", size=14, color=BRAND_ACCENT_ORANGE),
                bgcolor=BRAND_SECONDARY_BG_COLOR,
                padding=ft.padding.all(12),
            )
        self.page.update()

    # Removed update_log_subplot and all log/LAS code

    def add_point(self, _):
        """Add a new pressure point"""
        try:
            tvd = float(self.tvd_input.value or 0)
            pressure = float(self.pressure_input.value or 0)
            zone = self.zone_input.value or f"Point {len(self.points) + 1}"

            # Convert to base units if necessary
            current_depth_unit = self.depth_unit.value or "ft"
            current_pressure_unit = self.pressure_unit.value or "psi"

            tvd_base = convert_depth(tvd, current_depth_unit, "ft")
            pressure_base = convert_pressure(pressure, current_pressure_unit, "psi")

            self.points.append(
                PressurePoint(zone=zone, tvd=tvd_base, pressure=pressure_base)
            )
            self.shared_data["pressure_points"] = self.points
            self.update_table()
            self.update_plot()
            self.page.update()
        except (ValueError, TypeError):
            show_banner(
                self.page, "Invalid input. Please enter numeric values.", is_error=True
            )

    def save_data(self, _):
        """Save current data to file"""
        if save_points_to_file(self.points):
            show_banner(self.page, "Pressure points saved successfully.")
        else:
            show_banner(self.page, "Error saving pressure points.", is_error=True)

    def clear_table(self, _):
        """Clear all pressure points"""
        self.points.clear()
        self.shared_data["pressure_points"] = self.points
        self.update_table()
        self.update_plot()
        self.page.update()

    def manual_save_state(self, _):
        """Manual save button handler"""
        zoom_controls_dict = {
            "min": self.zoom_min_field.value,
            "max": self.zoom_max_field.value,
            "pressure_min": self.zoom_pressure_min_field.value,
            "pressure_max": self.zoom_pressure_max_field.value,
            "auto": self.auto_zoom_checkbox.value,
        }
        save_app_state(
            self.points,
            self.page,
            self.depth_unit.value or "ft",
            self.pressure_unit.value or "psi",
            zoom_controls_dict,
            self.curve_dropdown.value or "",
        )
        show_banner(self.page, "Application state saved.")

    def open_bulk_dialog(self, _):
        self.page.overlay.append(self.bulk_dialog)
        self.bulk_dialog.open = True
        self.page.update()

    def close_bulk_dialog(self, _=None):
        """Close the bulk dialog and refresh the page"""
        self.bulk_dialog.open = False
        self.page.update()
        logger.info("Bulk dialog closed and main app restored.")

    def process_bulk_data(self, _):
        """Process bulk input data"""
        try:
            tvd_lines = (
                self.tvd_area.value.strip().split("\n") if self.tvd_area.value else []
            )
            pressure_lines = (
                self.pressure_area.value.strip().split("\n")
                if self.pressure_area.value
                else []
            )
            zone_lines = (
                self.zone_area.value.strip().split("\n") if self.zone_area.value else []
            )
            if len(tvd_lines) != len(pressure_lines):
                show_banner(
                    self.page, "TVD and Pressure must have the same number of values"
                )
                return
            # If zone_lines is shorter, fill with empty strings
            if len(zone_lines) < len(tvd_lines):
                zone_lines += [""] * (len(tvd_lines) - len(zone_lines))
            new_points = []
            for tvd_str, pressure_str, zone_str in zip(
                tvd_lines, pressure_lines, zone_lines
            ):
                try:
                    tvd_val = float(tvd_str.strip())
                    pressure_val = float(pressure_str.strip())
                    zone_val = zone_str.strip()
                    internal_tvd = convert_depth(
                        tvd_val, self.depth_unit.value or "ft", "ft"
                    )
                    internal_pressure = convert_pressure(
                        pressure_val, self.pressure_unit.value or "psi", "psi"
                    )
                    new_points.append(
                        PressurePoint(
                            zone=zone_val, tvd=internal_tvd, pressure=internal_pressure
                        )
                    )
                except ValueError:
                    continue
            if new_points:
                self.points.extend(new_points)
                self.shared_data["pressure_points"] = self.points
                self.tvd_area.value = ""
                self.pressure_area.value = ""
                self.zone_area.value = ""
                self.update_table()
                self.update_plot()
                self.close_bulk_dialog()
                show_banner(self.page, f"Added {len(new_points)} points successfully!")
                self.page.update()
            else:
                show_banner(self.page, "No valid data found!")
        except (ValueError, TypeError, IndexError) as ex:
            logger.error("Error processing bulk data: %s", str(ex))
            show_banner(self.page, f"Error processing bulk data: {str(ex)}")

    def delete_point(self, e):
        index = e.control.data
        if 0 <= index < len(self.points):
            del self.points[index]
            self.update_table()
            self.update_plot()
            self.page.update()

    def toggle_point_selection(self, e):
        index = e.control.data
        if 0 <= index < len(self.points):
            self.points[index].selected = e.control.value
            self.update_plot()
            self.page.update()

    # Removed on_las_file_picked

    # Removed ensure_filepicker_and_pick

    def build(self):
        """Build and return the UI layout (no LAS/log controls)"""
        # Data controls above, plot below, both in a responsive row
        data_controls = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Data Input",
                        weight=ft.FontWeight.BOLD,
                        color=BRAND_DARK_GREEN,
                        size=13,
                    ),
                    ft.Row(
                        controls=[
                            self.zone_input,
                            self.tvd_input,
                        ],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        controls=[
                            self.pressure_input,
                            ft.IconButton(
                                icon=ft.Icons.ADD,
                                on_click=self.add_point,
                                bgcolor=BRAND_GREEN,
                                icon_color=BRAND_WHITE,
                                tooltip="Add Point",
                                style=ft.ButtonStyle(padding=ft.padding.all(2)),
                            ),
                        ],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        controls=[
                            ft.ElevatedButton(
                                "Bulk Add",
                                on_click=self.open_bulk_dialog,
                                bgcolor=BRAND_ACCENT_BLUE,
                                color=BRAND_WHITE,
                                style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=6, vertical=4)),
                            ),
                            ft.ElevatedButton(
                                "Save Points",
                                on_click=self.save_data,
                                bgcolor=BRAND_ACCENT_PURPLE,
                                color=BRAND_WHITE,
                                style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=6, vertical=4)),
                            ),
                            ft.ElevatedButton(
                                "Clear All Points",
                                on_click=self.clear_table,
                                bgcolor=BRAND_ACCENT_ORANGE,
                                color=BRAND_WHITE,
                                style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=6, vertical=4)),
                            ),
                        ],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    ft.Divider(),
                    ft.Text(
                        "Settings",
                        weight=ft.FontWeight.BOLD,
                        color=BRAND_DARK_GREEN,
                        size=13,
                    ),
                    ft.Row(
                        controls=[
                            self.depth_unit,
                            self.pressure_unit,
                            ft.ElevatedButton(
                                "Save App State",
                                on_click=self.manual_save_state,
                                bgcolor=BRAND_DARK_GRAY,
                                color=BRAND_WHITE,
                                style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=6, vertical=4)),
                            ),
                        ],
                        spacing=4,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                ],
                spacing=6,
            ),
            padding=2,
            border_radius=6,
            border=ft.border.all(1, BRAND_BORDER_COLOR),
            bgcolor=BRAND_SECONDARY_BG_COLOR,
            expand=True,
            width=320,
            margin=ft.margin.only(top=2, bottom=2),
        )

        plot_area = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Plot",
                        weight=ft.FontWeight.BOLD,
                        color=BRAND_DARK_GREEN,
                        size=18,
                    ),
                    self.plot_container,
                    ft.Row(
                        controls=[
                            self.zoom_min_field,
                            self.zoom_max_field,
                        ],
                        spacing=12,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    ft.Row(
                        controls=[
                            self.zoom_pressure_min_field,
                            self.zoom_pressure_max_field,
                        ],
                        spacing=12,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    ft.Row(
                        controls=[
                            self.auto_zoom_checkbox,
                            ft.ElevatedButton(
                                "Update Plot",
                                on_click=self.update_plot,
                                bgcolor=BRAND_ACCENT_BLUE,
                                color=BRAND_WHITE,
                                style=ft.ButtonStyle(padding=ft.padding.symmetric(horizontal=18, vertical=12)),
                            ),
                        ],
                        spacing=12,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                ],
                spacing=18,
            ),
            padding=12,
            border_radius=10,
            border=ft.border.all(1, BRAND_BORDER_COLOR),
            bgcolor=BRAND_SECONDARY_BG_COLOR,
            expand=True,
        )

        # Second row: left column (table + results), right column (plot)
        left_col = ft.Container(
            content=ft.Column([
                ft.Column([
                    self.table,
                    self.statistics_card
                ], spacing=16, expand=True)
            ], expand=True),
            border_radius=10,
            padding=12,
            col={"xs": 12, "sm": 12, "md": 12, "lg": 5, "xl": 5},
        )
        right_col = ft.Container(
            content=plot_area,
            border_radius=10,
            padding=12,
            col={"xs": 12, "sm": 12, "md": 12, "lg": 7, "xl": 7},
        )

        return ft.Column(
            expand=True,
            controls=[
                ft.ResponsiveRow(
                    controls=[data_controls],
                    spacing=16,
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Divider(),
                ft.ResponsiveRow(
                    controls=[left_col, right_col],
                    expand=True,
                    spacing=16,
                    alignment=ft.MainAxisAlignment.START,
                ),
            ],
            spacing=16,
        )



def create_pressure_gradient_app(page: ft.Page, shared_data: dict):
    """Main function for pressure gradient analysis page"""
    app = PressureGradientApp(page, shared_data)
    return app.build()


# =============================================================
# Flet Standalone App Entry Point
# =============================================================
def main(page: ft.Page):
    shared_data = {}
    page.title = "Pressure Gradient Analysis Toolkit"
    # Responsive: let Flet handle window size for mobile/tablet
    page.bgcolor = BRAND_BG_COLOR
    page.scroll = ft.ScrollMode.ALWAYS
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 10
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.START
    app_view = create_pressure_gradient_app(page, shared_data)
    page.add(app_view)


if __name__ == "__main__":
    # Use WEB_BROWSER view for deployment to Render.com
    ft.app(target=main, view=ft.WEB_BROWSER)
