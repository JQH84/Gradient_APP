# main.py (moved to project root for Render.com)

from src.main import *

# This file simply imports and runs the app from the src directory.
# Render.com will use this as the entry point.

if __name__ == "__main__":
    import flet as ft
    ft.app(target=main, view=ft.WEB_BROWSER)
